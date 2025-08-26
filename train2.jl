############################
# MQ-MMDRL (Lux Training API + AutoEnzyme)
############################

using ArgParse
using Lux, Optimisers, LuxCUDA, Lux.Training
using ReinforcementLearningTrajectories
import CommonRLInterface as CRL
using Gymnasium
using Random
using ComponentArrays
using ProgressMeter
using Wandb, Logging
using Statistics
using MLUtils
using Enzyme
# using CUDA
using ADTypes   # For AutoEnzyme

include("fast_replay.jl")

# --------- Small utils ----------
@inline dev() = gpu_device()
@inline todev(x) = dev()(x)

# Mean over quantiles to Q-values
function q_val(state, net, ps, st)
    ys, st = Lux.apply(net, state, ps, st)      # (N, na, B=1)
    q = dropdims(mean(ys; dims=1), dims=(1,3))  # (na,)
    return q, st
end

# Vectorized multiquadric kernel pieces
@inline function pairwise_multiquadric(X::AbstractMatrix{T}, Y::AbstractMatrix{T}; c::T = T(1)) where {T<:AbstractFloat}
    x2 = sum(abs2, X; dims=1)                # (1,Bx)
    y2 = sum(abs2, Y; dims=1)                # (1,By)
    XY = permutedims(X) * Y                  # (Bx,By)
    d2 = @. max(x2' + y2 - 2*XY, zero(T))
    return sqrt.(d2 .+ c^2)
end

@inline function mmd_loss_multiquadric(Ŷ::AbstractMatrix{T}, Y::AbstractMatrix{T}; c::T=T(1)) where {T<:AbstractFloat}
    Kxx = pairwise_multiquadric(Ŷ, Ŷ; c=c)
    Kyy = pairwise_multiquadric(Y,  Y;  c=c)
    Kxy = pairwise_multiquadric(Ŷ, Y;  c=c)
    return mean(Kxx) - 2T(1)*mean(Kxy) + mean(Kyy)
end

# Precompute τ grid once
@inline function make_cum_prob(N::Integer, T=Float32, devf=dev())
    return devf(range(T(1)/(2N), step=T(1)/N, length=N) |> collect)
end

# Quantile Huber (N,B) version
@inline function quantile_huber_loss(ŷ::AbstractMatrix{T}, y::AbstractMatrix{T}, cum_prob::AbstractVector{T}; κ::T=T(1)) where {T<:AbstractFloat}
    Δ = @views y .- ŷ                 # (N,B)
    abs_error = abs.(Δ)
    q = min.(abs_error, κ)
    huber = T(0.5) .* q .* q .+ κ .* (abs_error .- q)
    I = abs.(reshape(cum_prob, :, 1) .- (Δ .< 0))
    return mean(sum(I .* huber; dims=1))
end

"""
    select_actions(ys, actions) -> (N,B)

GPU-safe batched action selection (no scalar indexing).
- ys: (N, na, B)
- actions: Vector{Int} or CuArray{Int} length B
"""
function select_actions(ys::AbstractArray{T,3}, actions_in) where {T}
    N, na, B = size(ys)
    @assert length(actions_in) == B
    actions = isa(actions_in, CuArray) ? actions_in : cu(actions_in)
    j = cu(collect(1:na))                         # (na,) on GPU
    mask = reshape(j, na, 1) .== reshape(actions, 1, B)   # (na,B)
    sel = sum(ys .* reshape(mask, 1, na, B); dims=2)      # (N,1,B)
    return dropdims(sel; dims=2)                          # (N,B)
end

# ---------------- Objective Function ----------------
"""
Objective function for Lux Training API.

Must return (loss, new_st, stats).
`data` is a tuple: (state, target_ys, actions, cum_prob, λ).
"""
function obj_fn(model, ps, st, data)
    state, target_ys, actions, cum_prob, λ = data

    ys, st2 = Lux.apply(model, state, ps, st)
    ys_sel = select_actions(ys, actions)

    ql  = quantile_huber_loss(ys_sel, target_ys, cum_prob)
    mmd = mmd_loss_multiquadric(ys_sel, target_ys)

    loss = λ*ql + (1f0-λ)*mmd
    return loss, st2, (;)
end

# ---------------- Training ----------------
function train(game="breakout"; lambda::Float32=0.5f0)
    seed = 123
    rng = Random.default_rng(); Random.seed!(rng, seed)

    N = 50
    B = 32
    lr = 5f-5
    grad_clip = 10.0f0
    n_step = 1

    # Envs
    env = atari_env(game; terminal_on_life_loss=false, frame_skip=4, repeat_action_probability=0.0, seed)
    eval_env = atari_env(game; terminal_on_life_loss=false, frame_skip=4, repeat_action_probability=0.0, seed)
    na = length(CRL.actions(env))

    # Net
    net = Lux.Chain(
        Lux.Conv((8,8), 4=>32, relu; pad=0, stride=4),
        Lux.Conv((4,4), 32=>64, relu; pad=0, stride=2),
        Lux.Conv((3,3), 64=>64, relu; pad=0, stride=1),
        Lux.FlattenLayer(),
        Lux.Dense(7*7*64, 512, relu),
        Lux.Parallel(
            (x,z)->(x .+ unsqueeze(z; dims=1)),
            Lux.Chain(
                Lux.Dense(512, na*N),
                x -> reshape(x, N, na, :),
                relu,
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

    buffer = Trajectory(
        CircularArraySARTSTraces(; capacity=100_000,
            state=Float32 => (84,84,4),
            action=Int => (),
            reward=Float32 => (),
            terminal=Bool => ()),
        BatchSampler{SS′ART}(batchsize=B, rng=rng),
        InsertSampleRatioController(; ratio=0.25f0, threshold=1),
    )

    cum_prob = make_cum_prob(N)

    T = 50_000_000
    p = Progress(T); t = 0
    eps_min = 0.01; eps_decay = 250_000

    lg = WandbLogger(; project="ContQuant", name="$(game)")
    update_config!(lg, Dict("N"=>N,"B"=>B,"lr"=>lr,"grad_clip"=>grad_clip,
                            "loss_func"=>"quantile_huber+mmd","seed"=>seed,"lambda"=>lambda))

    backend = AutoEnzyme()   # could also use AutoZygote(), AutoReactant() later

    while t < T
        CRL.reset!(env)
        state = CRL.observe(env)
        push!(buffer, (; state))
        l = 0; approx_loss = 0f0; total_reward = 0f0

        while !CRL.terminated(env)
            l += 1
            eps = max(eps_min, 1 - (1 - eps_min) * t / eps_decay)
            action = if rand() < eps
                rand(CRL.actions(env))
            else
                sdev = todev(unsqueeze(CRL.observe(env), dims=4)) .* (1f0/255)
                q, st = q_val(sdev, ts.model, ts.parameters, ts.states)
                CRL.actions(env)[argmax(q)]
            end

            reward = CRL.act!(env, action)
            total_reward += reward
            state = CRL.observe(env)
            push!(buffer, (; action=action, reward, state, terminal=CRL.terminated(env)))

            for batch in buffer
                t += 1

                # ---------- Target ----------
                bs = todev(batch.next_state) .* (1f0/255)      # (84,84,4,B)
                ys1, st = Lux.apply(ts.model, bs, target_ps, ts.states)
                q = dropdims(mean(ys1; dims=1), dims=1)        # (na,B)
                a = vec(argmax(q; dims=1))                     # (B,)
                ys1_sel = select_actions(ys1, a)               # (N,B)

                γ = 0.99f0^n_step
                not_done = todev(1f0 .- f32(batch.terminal))   # (B,)
                r        = todev(f32(batch.reward))            # (B,)

                target_ys = γ .* (ys1_sel .* reshape(not_done,1,B)) .+ reshape(r,1,B)

                # ---------- Training Step ----------
                sdev = todev(batch.state) .* (1f0/255)
                data = (sdev, target_ys, batch.action, cum_prob, lambda)

                grads, loss_val, stats, ts = Training.single_train_step!(backend, obj_fn, data, ts)

                approx_loss += loss_val

                # target update
                if t % 10_000 == 0
                    target_ps .= ts.parameters
                end

                if t % 10_000 == 0
                    with_logger(lg) do
                        @info "train" step=t loss=(approx_loss/10_000)
                    end
                    approx_loss = 0f0
                end

                if t % 1_000_000 == 0
                    rewards = Float32[]
                    for _ in 1:10
                        CRL.reset!(eval_env)
                        trew=0f0; stepi=0
                        while !CRL.terminated(eval_env) && stepi < 10_000
                            stepi += 1
                            sdev = todev(unsqueeze(CRL.observe(eval_env), dims=4)) .* (1f0/255)
                            q, st = q_val(sdev, ts.model, ts.parameters, ts.states)
                            ai = argmax(q)
                            trew += CRL.act!(eval_env, CRL.actions(eval_env)[ai])
                        end
                        push!(rewards, trew)
                    end
                    with_logger(lg) do
                        @info "eval" mean_reward=mean(rewards) max_reward=maximum(rewards) min_reward=minimum(rewards)
                    end
                end

                if t % 5_000 == 0
                    ProgressMeter.update!(p, t; showvalues=[("step",t)])
                end
            end
        end

        with_logger(lg) do
            @info "episode" reward=total_reward steps=t len=l
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
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    train("breakout"; lambda=0.5f0)
    # train(args["game"]; lambda=args["lambda"])
end

main()
