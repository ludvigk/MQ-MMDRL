using ArgParse
using Lux, Optimisers
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

function q_val(state, net, ps, st)
    ys, st = Lux.apply(net, state, ps, st)
    return mean(ys, dims = 1) |> vec, st
end

function q_val(ys)
    # segm_int = dx .* (ys[1:end-1,:,:] .+ ys[2:end, :,:]) ./ 2
    # dropdims(sum(segm_int; dims=1); dims=1)
    return dropdims(mean(ys, dims = 1), dims = 1)
end

function multiquadric_kernel(x, y; c = 1.0f0)
    # The multiquadric kernel is defined as sqrt(‖x - y‖² + c²)
    return sqrt(sum(abs2, x .- y) + c^2)
end

function mmd_loss_multiquadric(ŷ, y; c = 1.0f0)
    # MMD loss is calculated as:
    # MMD²(X, Y) = E[k(X, X')] - 2E[k(X, Y)] + E[k(Y, Y')]
    
    # Calculate the kernel matrices
    K_ŷŷ = [multiquadric_kernel(ŷ[:, i], ŷ[:, j]; c=c) for i in 1:size(ŷ, 2), j in 1:size(ŷ, 2)]
    K_yy = [multiquadric_kernel(y[:, i], y[:, j]; c=c) for i in 1:size(y, 2), j in 1:size(y, 2)]
    K_ŷy = [multiquadric_kernel(ŷ[:, i], y[:, j]; c=c) for i in 1:size(ŷ, 2), j in 1:size(y, 2)]
    
    # Calculate the mean of the kernel matrices
    mmd2 = (mean(K_ŷŷ) - 2 * mean(K_ŷy) + mean(K_yy))
    
    return mmd2
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

function train(game="breakout"; lambda::Float32=0.5f0)
    seed = 123
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    N = 200
    B = 32
    lr = 0.00005
    grad_clip = 10.0
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
            capacity = 1_000_000,
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
        mean_q = []
        GC.gc()
        while !CRL.terminated(env)
            l += 1
            eps_threshold = max(eps_min, 1 - (1 - eps_min) * t / eps_decay)
            if rand() < eps_threshold
                action = CRL.actions(env) |> rand
            else
                state = (unsqueeze(CRL.observe(env), dims = 4) ./ 255.0f0) |> gdev
                q, st = q_val(state, net, ps, st)
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
                a = dropdims(argmax(q; dims = 1); dims = 1)

                @views ys1 = ys1[:, a]

                target_ys = 0.99f0^n_step .* ys1 .* gdev(reshape(1 .- batch.terminal, 1, B)) .+ gdev(reshape(batch.reward, 1, B))
                a_ind = CartesianIndex.(batch.action, 1:length(batch.action))

                state = (batch.state ./ 255.0f0) |> gdev
                res = Zygote.withgradient(ps) do ps
                    ys2, st = Lux.apply(net, state, ps, st)
                    ys2 = ys2[:, a_ind]
                    loss = lambda * quantile_huber_loss(ys2, target_ys)
                    loss += (1 - lambda) * mmd_loss_multiquadric(ys2, target_ys)
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
        if isempty(mean_q)
            mean_q = [0]
        end
        with_logger(lg) do
            @info "metrics" loss = approx_loss reward = total_reward t = t l = l mean_q_val = mean(mean_q)
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