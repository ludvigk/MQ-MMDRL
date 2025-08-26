using ArgParse
using Lux, Optimisers, LuxCUDA
using ReinforcementLearningTrajectories
import CommonRLInterface as CRL
using Gymnasium
using Random
using ComponentArrays
using ProgressMeter
using Wandb, Logging
using Statistics
using MLUtils
using ChainRulesCore
using Zygote
using NNlib: gather


function q_val(state, net, ps, st)
    ys, st = Lux.apply(net, state, ps, st)
    return mean(ys, dims = 1) |> vec, st
end

function q_val(ys)
    # segm_int = dx .* (ys[1:end-1,:,:] .+ ys[2:end, :,:]) ./ 2
    # dropdims(sum(segm_int; dims=1); dims=1)
    return dropdims(mean(ys, dims = 1), dims = 1)
end

# Avoid scalar loops; work with (N, B) matrices directly.
# Uses the identity ||xi - yj||^2 = ||xi||^2 + ||yj||^2 - 2 xi⋅yj
@inline function pairwise_multiquadric(X::AbstractMatrix{T}, Y::AbstractMatrix{T}; c::T = T(1)) where {T<:AbstractFloat}
    # X, Y: (N, Bx) and (N, By)
    # returns (Bx, By)
    x2 = sum(abs2, X; dims=1)               # (1, Bx)
    y2 = sum(abs2, Y; dims=1)               # (1, By)
    XY = permutedims(X) * Y                 # (Bx, By)
    d2 = @. max(x2' + y2 - 2*XY, zero(T))   # numerical safety
    return sqrt.(d2 .+ c^2)
end

@inline function mmd_loss_multiquadric(Ŷ::AbstractMatrix{T}, Y::AbstractMatrix{T}; c::T = T(1)) where {T<:AbstractFloat}
    Kxx = pairwise_multiquadric(Ŷ, Ŷ; c=c)
    Kyy = pairwise_multiquadric(Y,  Y;  c=c)
    Kxy = pairwise_multiquadric(Ŷ, Y;  c=c)
    return mean(Kxx) - 2T(1)*mean(Kxy) + mean(Kyy)
end

function quantile_huber_loss(ŷ, y; κ = 1.0f0)
    N, B = size(y)
    Δ = reshape(y, N, 1, B) .- reshape(ŷ, 1, N, B)
    abs_error = abs.(Δ)
    quadratic = min.(abs_error, κ)
    linear = abs_error .- quadratic
    huber_loss = 0.5f0 .* quadratic .* quadratic .+ κ .* linear

    cum_prob = ignore_derivatives() do
        StepRangeLen(0.5f0 / N, 1.0f0 / N, N) |> collect |> gpu_device()
    end

    loss = ignore_derivatives(abs.(cum_prob .- (Δ .< 0))) .* huber_loss
    return mean(sum(loss; dims = 1))
end

# Build this ONCE outside the loop and pass it in.
@inline function make_cum_prob(N::Integer, T=Float32, dev=gpu_device())
    return dev(range(T(1)/(2N), step=T(1)/N, length=N) |> collect)
end

@inline function quantile_huber_loss(ŷ::AbstractMatrix{T}, y::AbstractMatrix{T}, cum_prob::AbstractVector{T}; κ::T = T(1)) where {T<:AbstractFloat}
    # ŷ, y: (N, B); cum_prob: (N,)
    Δ = @views y .- ŷ                       # (N, B)
    abs_error = abs.(Δ)
    q = min.(abs_error, κ)
    huber = T(0.5) .* q .* q .+ κ .* (abs_error .- q)

    # Broadcast cum_prob across batch (N,1) vs (N,B)
    # indicator(Δ < 0) is non-differentiable; keep as Bool → Float
    I = @. abs(cum_prob - (Δ < 0))
    return mean(sum(I .* huber; dims=1))
end

function select_actions(ys::CuArray{T,3}, actions::CuArray{Int}) where {T}
    N, na, B = size(ys)

    # Turn actions into a (1,B) array, then broadcast to shape (N,B)
    idx_na = reshape(actions, 1, B) |> gpu_device()
    idx_na = repeat(idx_na, N, 1)  # (N,B)

    # Row indices 1:N, repeated across batch
    idx_n = repeat(reshape(1:N, N, 1), 1, B) |> gpu_device()

    # Batch indices 1:B, repeated across N
    idx_b = repeat(reshape(1:B, 1, B), N, 1) |> gpu_device()

    # Now gather wants a tuple of index arrays, all same shape
    return gather(ys, (idx_n, idx_na, idx_b))  # (N,B)
end


function train(game="breakout"; lambda::Float32=0.5f0)
    seed = 123
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    N = 50
    B = 32
    lr = 0.00005f0
    grad_clip = 10.0f0
    ρ = 0.005f0
    n_step = 1
    EVAL_EVERY = 250_000

    gdev = gpu_device()

    env = atari_env(
        game;
        terminal_on_life_loss = false,
        frame_skip = 4,
        repeat_action_probability = 0.0,
        seed
    )
    eval_env = atari_env(
        game;
        terminal_on_life_loss = false,
        frame_skip = 4,
        repeat_action_probability = 0.0,
        seed
    )

    na = CRL.actions(env) |> length

    net = Lux.Chain(
        Lux.Conv((8, 8), 4 => 32, relu; pad = 0, stride = 4),
        Lux.Conv((4, 4), 32 => 64, relu; pad = 0, stride = 2),
        Lux.Conv((3, 3), 64 => 64, relu; pad = 0, stride = 1),
        Lux.FlattenLayer(),
        Lux.Dense(7 * 7 * 64, 512, relu),
        Lux.Parallel(
            (x, z) -> begin
                b = unsqueeze(z; dims = 1)
                x .+ b
            end,
            Lux.Chain(
                Lux.Dense(512, na * N),
                x -> reshape(x, N, na, :),
                relu,
                x -> 1.0f-3 .+ x,
                x -> cumsum(x; dims = 1),
            ),
            Lux.Dense(512, na),
        ),
    )

    ps, st = Lux.setup(rng, net)
    ps = ComponentArray(ps) |> gdev
    target_ps = copy(ps) |> gdev

    opt = Optimisers.OptimiserChain(ClipGrad(grad_clip), Optimisers.Adam(lr, (0.9, 0.999), 1.0f-2 / 32))
    opt_st = Optimisers.setup(opt, ps)

    buffer = Trajectory(
        CircularArraySARTSTraces(;
            capacity = 100_000,
            state = Float32 => (84, 84, 4),
            action = Int => (),
            reward = Float32 => (),
            terminal = Bool => (),
        ),
        BatchSampler{SS′ART}(
            batchsize = B,
            rng = rng
        ),
        InsertSampleRatioController(; ratio = 0.25f0, threshold = 1),
    )

    T = 50_000_000
    p = Progress(T)
    t = 0
    eps_min = 0.01
    eps_decay = 250_000

    lg = WandbLogger(; project = "ContQuant", name = "$(game)")
    update_config!(
        lg, Dict(
            "N" => N, "B" => B, "lr" => lr, "grad_clip" => grad_clip,
            "rho" => ρ, "loss_func" => "quantile_huber_loss", "seed" => seed,
            "lambda" => lambda,
        )
    )

    while t < T
        CRL.reset!(env)
        state = CRL.observe(env)
        push!(buffer, (; state))
        l = 0
        approx_loss = 0
        total_reward = 0
        while !CRL.terminated(env)
            l += 1
            eps_threshold = max(eps_min, 1 - (1 - eps_min) * t / eps_decay)
            if rand() < eps_threshold
                action = CRL.actions(env) |> rand
            else
                # state = (unsqueeze(CRL.observe(env), dims = 4) ./ 255.0f0) |> gdev
                sdev = gpu_device()(unsqueeze(CRL.observe(env), dims=4)) .* (1f0/255)

                q, st = q_val(sdev, net, ps, st)
                # q, st = Lux.apply(net, state, ps, st)
                ai = argmax(q |> vec)
                action = CRL.actions(env)[ai]
            end
            reward = CRL.act!(env, action)
            total_reward += reward
            state = CRL.observe(env)
            push!(buffer, (; action = action, reward, state, terminal = CRL.terminated(env)))

            for batch in buffer
                t += 1

                bs = (batch.next_state ./ 255.0f0) |> gdev
                ys1, st = Lux.apply(net, bs, target_ps, st)
                q = dropdims(mean(ys1, dims = 1), dims = 1)
                a = getindex.(vec(argmax(q; dims = 1)), 1)

                # @views ys1 = ys1[:, a]
                ys1_sel = select_actions(ys1, a)        # (N, B)
                target_ys = 0.99f0^n_step .* ys1_sel .* gdev(reshape(1 .- batch.terminal, 1, B)) .+ gdev(reshape(batch.reward, 1, B))
                # a_ind = CartesianIndex.(batch.action, 1:length(batch.action))

                state = (batch.state ./ 255.0f0) |> gdev
                res = Zygote.withgradient(ps) do ps
                    ys2, st = Lux.apply(net, state, ps, st)
                    # ys2 = ys2[:, a_ind]
                    ys2_sel = select_actions(ys2, batch.action)  # (N, B)
                    loss = lambda * quantile_huber_loss(ys2_sel, target_ys)
                    loss += (1 - lambda) * mmd_loss_multiquadric(ys2_sel, target_ys)
                    loss
                end
                grads = res.grad |> only
                opt_st, ps = Optimisers.update!(opt_st, ps, grads)
                # target_ps .= target_ps .* (1 - ρ) .+ ρ .* ps
                approx_loss += res.val
                if t % 10_000 == 0
                    target_ps .= ps
                end
                if t % EVAL_EVERY == 0
                    test_t = 0
                    rewards = Vector{Float32}()
                    while test_t < 125_000
                        CRL.reset!(eval_env)
                        total_reward = 0
                        l = 0
                        while !CRL.terminated(eval_env)
                            l += 1
                            test_t += 1
                            state = (unsqueeze(CRL.observe(eval_env), dims = 4) ./ 255.0f0) |> gdev
                            q, st = q_val(state, net, ps, st)
                            ai = argmax(q |> vec)
                            action = CRL.actions(env)[ai]
                            reward = CRL.act!(eval_env, action)
                            total_reward += reward
                        end
                        push!(rewards, total_reward)
                    end
                    mean_reward = mean(rewards[1:(end - 1)])
                    max_reward = maximum(rewards[1:(end - 1)])
                    min_reward = minimum(rewards[1:(end - 1)])
                    with_logger(lg) do
                        @info "test" mean_reward max_reward min_reward
                    end
                end
            end
        end
        approx_loss = approx_loss / l
        with_logger(lg) do
            @info "metrics" loss = approx_loss reward = total_reward t = t l = l
        end
        ProgressMeter.update!(p, t; showvalues = [("step", t), ("score", total_reward), ("loss", round(approx_loss; digits = 3))])
    end
    close(lg)
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--lambda", "-l"
            help = "lambda"
            arg_type = Float32
            default = 0.5
        "--game", "-g"
            help = "game"
            arg_type = String
            default = "breakout"
    end
    return parse_args(s)
end


function main()
    parsed_args = parse_commandline()
    train(parsed_args["game"]; lambda=parsed_args["lambda"])
end

main()