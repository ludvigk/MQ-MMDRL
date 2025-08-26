############################
# train2_fastreplay.jl  —  Multi-Env + Threaded Rollouts
############################

using ArgParse
using Lux, Optimisers, LuxCUDA, Lux.Training
import CommonRLInterface as CRL
using Random
using Gymnasium
using ComponentArrays
using ProgressMeter
using Wandb, Logging
using Statistics
using Zygote
using ADTypes
using NNlib
using MLUtils
using Base.Threads

include("fast_replay2.jl")   # FastReplay with sample_batch!

# ---------- Small utils ----------
@inline dev() = gpu_device()
@inline todev(x) = dev()(x)

# Mean over quantiles to Q-values
function q_val_batch(states4d, net, ps, st)
    ys, st = Lux.apply(net, states4d, ps, st)  # (N, na, B=E)
    q = dropdims(mean(ys; dims=1), dims=1) # (na, E)
    return q, st
end

# GPU-safe batched action selection for a batch of Qs and an action set
# q: (na, E), actions(env): 1..na
@inline greedy_actions(q::AbstractMatrix) = vec(argmax(q; dims=1))

# ---------- Objective ----------
@inline function pairwise_multiquadric(X::AbstractMatrix{T}, Y::AbstractMatrix{T}; c::T = T(1)) where {T<:AbstractFloat}
    # x2 = sum(abs2, X; dims=1)
    # y2 = sum(abs2, Y; dims=1)
    # XY = permutedims(X) * Y                  # (Bx,By)
    # d2 = x2' .+ y2 .- 2*XY
    d2 = (unsqueeze(X, dims=1) .- unsqueeze(Y, dims=2)) .^ 2
    return -sqrt.(d2 .+ c^2)
end

@inline function mmd_loss_multiquadric(Ŷ::AbstractMatrix{T}, Y::AbstractMatrix{T}; c::T=T(1)) where {T<:AbstractFloat}
    Kxx = pairwise_multiquadric(Ŷ, Ŷ; c=c)
    Kyy = pairwise_multiquadric(Y, Y;  c=c)
    Kxy = pairwise_multiquadric(Ŷ, Y;  c=c)
    return mean(Kxx) - 2T(1)*mean(Kxy) + mean(Kyy)
end

@inline function make_cum_prob(N::Integer, T=Float32, devf=dev())
    return devf(range(T(1)/(2N), step=T(1)/N, length=N) |> collect)
end

@inline function quantile_huber_loss(ŷ::AbstractMatrix{T}, y::AbstractMatrix{T}, cum_prob::AbstractVector{T}; κ::T=T(1)) where {T<:AbstractFloat}
    Δ = @views y .- ŷ
    abs_error = abs.(Δ)
    q = min.(abs_error, κ)
    huber = T(0.5) .* q .* q .+ κ .* (abs_error .- q)
    I = abs.(reshape(cum_prob, :, 1) .- (Δ .< 0))
    return mean(sum(I .* huber; dims=1))
end

# Select actions (N,na,B) x (B,) -> (N,B)
function select_actions(ys::AbstractArray{T,3}, actions_in) where {T}
    N, na, B = size(ys)
    @assert length(actions_in) == B
    actions = isa(actions_in, CuArray) ? actions_in : cu(actions_in)
    j = cu(collect(1:na))
    mask = reshape(j, na, 1) .== reshape(actions, 1, B)   # (na,B)
    sel = sum(ys .* reshape(mask, 1, na, B); dims=2)
    return dropdims(sel; dims=2)                           # (N,B)
end

function obj_fn(model, ps, st, data)
    state, target_ys, actions, cum_prob, λ = data
    ys, st2 = Lux.apply(model, state, ps, st)
    ys_sel = select_actions(ys, actions)
    ql  = quantile_huber_loss(ys_sel, target_ys, cum_prob)
    mmd = mmd_loss_multiquadric(ys_sel, target_ys)
    loss = λ*ql + (1f0-λ)*mmd
    return loss, st2, (;)
end

# ---------- Multi-env helpers ----------
struct VecEnv
    envs::Vector
    na::Int
end

function make_envs(game::String, E::Int; seed=123, frame_skip=4, repeat_action_probability=0.25, terminal_on_life_loss=false)
    envs = Vector{Any}(undef, E)
    for i in 1:E
        envs[i] = atari_env(game; terminal_on_life_loss, frame_skip, repeat_action_probability, seed=seed+i)
    end
    VecEnv(envs, length(CRL.actions(envs[1])))
end

# Reset a subset of envs (mask) and write first frame into frame buffer
function reset_masked!(VE::VecEnv, mask::Vector{Bool}, frames::Array{UInt8,4})
    @threads for i in 1:length(VE.envs)
        if mask[i]
            CRL.reset!(VE.envs[i])
            f = CRL.observe(VE.envs[i])              # (84,84,4)
            @views frames[:,:,:,i] .= f
        end
    end
end

# Step all envs in parallel given actions; fill rewards/dones/next frames
function step_vec!(VE::VecEnv, actions::Vector{Int}, rewards::Vector{Float32},
                   dones::Vector{Bool}, frames::Array{UInt8,4})
    @threads for i in 1:length(VE.envs)
        r = CRL.act!(VE.envs[i], actions[i])
        rewards[i] = r
        dones[i] = CRL.terminated(VE.envs[i])
        f = CRL.observe(VE.envs[i])
        @views frames[:,:,:,i] .= f
    end
end

# ---------- Training ----------
function train(game::String="breakout"; lambda::Float32=0.5f0)
    seed = 123
    rng = Random.default_rng(); Random.seed!(rng, seed)

    # ---- knobs ----
    E  = min(nthreads(), 16)  # number of parallel envs
    Nq = 200                  # quantiles
    B  = 32                   # learner batch size (independent of E)
    lr = 5f-5
    grad_clip = 10.0f0
    n_step = 1
    γ = 0.99f0^n_step

    # Make envs
    VE = make_envs(game, E; seed, frame_skip=4, repeat_action_probability=0.0, terminal_on_life_loss=false)
    na = VE.na

    # Net (Conv+ReLU fused; AutoZygote handles cuDNN fine)
    net = Lux.Chain(
        Lux.Conv((8,8), 4=>32, Lux.relu; pad=0, stride=4),
        Lux.Conv((4,4), 32=>64, Lux.relu; pad=0, stride=2),
        Lux.Conv((3,3), 64=>64, Lux.relu; pad=0, stride=1),
        Lux.FlattenLayer(),
        Lux.Dense(7*7*64, 512, Lux.relu),
        Lux.Parallel(
            (x, z)->(x .+ unsqueeze(z; dims=1)),
            Lux.Chain(
                Lux.Dense(512, na*Nq, Lux.relu),
                x -> reshape(x, Nq, na, :),
                x -> 1.0f-3 .+ x,
                x -> cumsum(x; dims=1),
            ),
            Lux.Dense(512, na),
        ),
    )
    ps, st = Lux.setup(rng, net)
    ps = ComponentArray(ps) |> todev
    target_ps = copy(ps)

    opt = Optimisers.OptimiserChain(ClipGrad(grad_clip),
                                    Optimisers.Adam(lr, (0.9,0.999), 1.0f-2/32))
    ts = Lux.Training.TrainState(net, ps, st, opt)

    # Replay (UInt8 frames)
    rb = FastReplay.ReplayBuffer(84, 84; n_frames=4, capacity=1_000_000)

    # Exploration & schedule
    WARMUP_FRAMES = 500
    eps_min = 0.01; eps_decay_frames = 25_000
    function eps(frames_done::Int)
        return max(eps_min, 1 - (1 - eps_min) * frames_done / eps_decay_frames)
    end

    cum_prob = make_cum_prob(Nq)

    # Progress
    T_frames = 50_000_000
    p = Progress(T_frames)
    frames_done = 0

    lg = WandbLogger(; project="ContQuant", name="$(game)")
    update_config!(lg, Dict("N"=>Nq,"B"=>B,"lr"=>lr,"grad_clip"=>grad_clip,
                            "loss_func"=>"quantile_huber+mmd","seed"=>seed,"lambda"=>lambda,"E"=>E))

    backend = AutoZygote()

    # ---------- Vec buffers ----------
    # Per-env rollout buffers
    f_u8 = Array{UInt8}(undef, 84,84,4,E)     # current stacked obs (UInt8) for each env
    reset_mask = fill(true, E)                # reset all envs initially
    reset_masked!(VE, reset_mask, f_u8)

    # Per-env transient arrays
    rew = Vector{Float32}(undef, E)
    dn  = Vector{Bool}(undef, E)
    acts= Vector{Int}(undef, E)

    # Prealloc learner batch on host
    s_u8   = Array{UInt8}(undef, 84,84,4,B)
    sp_u8  = Array{UInt8}(undef, 84,84,4,B)
    next_m = BitVector(undef, B)
    acts_b = Vector{Int16}(undef, B)
    rews_b = Vector{Float32}(undef, B)
    dones_b= BitVector(undef, B)
    idxs_b = Vector{Int}(undef, B)

    # ---------- main loop ----------
    while frames_done < T_frames
        # ε-greedy for all envs at once (vectorized)
        ε = eps(frames_done)
        # Push current frames to device and pick greedy actions in batch
        sdev = todev(f_u8) .* (1f0/255)           # (84,84,4,E)
        Q, st = q_val_batch(sdev, ts.model, ts.parameters, ts.states)  # (na,E)
        greedy = greedy_actions(Q) |> cpu_device()                     # (E,)

        # Mix ε-greedy
        for i in 1:E
            acts[i] = (rand() < ε || frames_done < WARMUP_FRAMES) ?
                       rand(1:na) : greedy[i][1]
        end
        # Map 1..na to actual environment action indices
        # CommonRLInterface uses Base.OneTo, so CRL.actions(env)[k] yields env action
        @threads for i in 1:E
            acts[i] = CRL.actions(VE.envs[i])[acts[i]]
        end

        # Step all envs in parallel; fill next frames and rewards/dones
        step_vec!(VE, acts, rew, dn, f_u8)

        # Push transitions into replay (only last frame of stack)
        @threads for i in 1:E
            @views FastReplay.push!(rb, f_u8[:,:,end,i], acts[i], rew[i], dn[i])
        end

        frames_done += E

        # Reset any finished envs and seed a frame so stacks remain continuous
        fill!(reset_mask, false)
        for i in 1:E
            if dn[i]
                reset_mask[i] = true
            end
        end
        if any(reset_mask)
            reset_masked!(VE, reset_mask, f_u8)
            # after reset, also push a seed frame per reset env (action=0,reward=0,done=false)
            @threads for i in 1:E
                if reset_mask[i]
                    @views FastReplay.push!(rb, f_u8[:,:,end,i], 0, 0f0, false)
                end
            end
            frames_done += count(reset_mask)  # counting these seed frames keeps schedule consistent
        end

        # ---------- Learner updates ----------
        if rb.size >= max(WARMUP_FRAMES, 4B)
            # Do k updates per rollout step, e.g., k = round(Int, B / E) to keep pace
            # K = max(1, div(B, E))
            K = 2
            for _ in 1:K
                FastReplay.sample_batch!(rb, s_u8, sp_u8, next_m, acts_b, rews_b, dones_b, idxs_b; rng=rng)

                # H2D + scale on device
                bs  = todev(s_u8)  .* (1f0/255)
                bsp = todev(sp_u8) .* (1f0/255)

                # Target network
                ys1, st = Lux.apply(ts.model, bsp, target_ps, ts.states)   # (Nq,na,B)
                q = dropdims(mean(ys1; dims=1), dims=1)                     # (na,B)
                a = vec(argmax(q; dims=1))                                   # (B,)
                ys1_sel = select_actions(ys1, a)                             # (Nq,B)

                not_done = todev(Float32.(next_m))                           # (B,)
                r        = todev(rews_b)                                     # (B,)
                target_ys = γ .* (ys1_sel .* reshape(not_done,1,B)) .+ reshape(r,1,B)

                data = (bs, target_ys, acts_b, cum_prob, lambda)
                grads, loss_val, stats, ts = Training.single_train_step!(backend, obj_fn, data, ts)

                if frames_done % 10_000 < E
                    target_ps .= ts.parameters
                end
                if frames_done % 5_000 < E
                    ProgressMeter.update!(p, frames_done; showvalues=[("frames",frames_done), ("ε",round(ε,digits=3))])
                end
                if frames_done % 10_000 < E
                    with_logger(lg) do
                        @info "train" frames=frames_done loss=loss_val
                    end
                end
            end
        elseif frames_done % 5_000 < E
            ProgressMeter.update!(p, frames_done; showvalues=[("frames",frames_done), ("ε",round(eps(frames_done),digits=3))])
        end

        # ------ (Optional) periodic eval ------
        if frames_done % 250_000 < E
            rewards = Float32[]
            t_eval = 0
            env_eval = atari_env(game; terminal_on_life_loss=false, frame_skip=4, repeat_action_probability=0.0, seed=seed+9999)
            while t_eval < 125_000
                CRL.reset!(env_eval)
                trew=0f0
                while !CRL.terminated(env_eval)
                    t_eval += 1
                    s = CRL.observe(env_eval)
                    sdev = todev(reshape(s, 84,84,4,1)) .* (1f0/255)
                    q,_ = q_val_batch(sdev, ts.model, ts.parameters, ts.states)
                    ai = argmax(q)
                    trew += CRL.act!(env_eval, CRL.actions(env_eval)[ai])
                end
                push!(rewards, trew)
            end
            with_logger(lg) do
                @info "eval" frames=frames_done mean_reward=mean(rewards) max_reward=maximum(rewards) min_reward=minimum(rewards)
            end
        end
    end
    close(lg)
end

# ------------- CLI -------------
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--lambda" "-l"
            help="lambda"
            arg_type=Float32
            default=0.5
        "--game" "-g"
            help="game"
            arg_type=String
            default="breakout"
        "--envs" "-e"
            help="number of parallel envs"
            arg_type=Int
            default=min(nthreads(), 16)
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    train("breakout"; lambda=0.5f0)
end

main()
