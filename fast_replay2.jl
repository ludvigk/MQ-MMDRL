module FastReplay

using Random

export ReplayBuffer, sample_batch, sample_batch!, push!

# ------------ Ring buffer storing raw UInt8 frames ------------
mutable struct ReplayBuffer
    H::Int
    W::Int
    n_frames::Int
    capacity::Int

    frames::Array{UInt8,3}      # (H, W, capacity) grayscale
    actions::Vector{Int16}
    rewards::Vector{Float32}
    dones::BitVector

    episode_id::Vector{Int32}
    fsr::Vector{Int32}          # frames_since_reset for O(1) validity

    size::Int
    write_pos::Int
    current_episode::Int32
end

function ReplayBuffer(H::Int, W::Int; n_frames::Int=4, capacity::Int=1_000_000)
    ReplayBuffer(
        H, W, n_frames, capacity,
        Array{UInt8}(undef, H, W, capacity),
        Vector{Int16}(undef, capacity),
        Vector{Float32}(undef, capacity),
        falses(capacity),
        Vector{Int32}(undef, capacity),
        Vector{Int32}(undef, capacity),
        0, 1, Int32(1)
    )
end

@inline _incpos(pos::Int, cap::Int) = (pos == cap ? 1 : pos + 1)

"""
    push!(rb, frame::AbstractArray{UInt8,2}, action, reward, done)

Push a SINGLE frame (H,W) and its transition data.
"""
function Base.push!(rb::ReplayBuffer, frame::AbstractArray{UInt8,2}, action, reward, done::Bool)
    @inbounds rb.frames[:,:,rb.write_pos] = frame
    @inbounds rb.actions[rb.write_pos] = Int16(action)
    @inbounds rb.rewards[rb.write_pos] = Float32(reward)
    @inbounds rb.dones[rb.write_pos] = done

    prev = (rb.size == 0) ? rb.write_pos :
           (rb.write_pos == 1 ? (rb.size == rb.capacity ? rb.capacity : rb.size) : rb.write_pos - 1)

    if rb.size == 0
        rb.episode_id[rb.write_pos] = rb.current_episode
        rb.fsr[rb.write_pos] = 1
    else
        rb.episode_id[rb.write_pos] = rb.current_episode
        rb.fsr[rb.write_pos] = rb.dones[prev] ? 1 : rb.fsr[prev] + 1
    end

    if done
        rb.current_episode += 1
    end

    if rb.size < rb.capacity
        rb.size += 1
    end
    rb.write_pos = _incpos(rb.write_pos, rb.capacity)
    return rb
end

@inline valid_state_at(rb::ReplayBuffer, i::Int) = (rb.fsr[i] >= rb.n_frames)

function sample_index(rb::ReplayBuffer, rng::AbstractRNG)
    @assert rb.size >= rb.n_frames "Not enough frames to sample yet."
    hi = rb.size
    while true
        i = rand(rng, 1:hi)
        valid_state_at(rb, i) && return i
    end
end

# ------------ fast copy helper (accepts Array or SubArray dest) ------------
@inline function copy_stack!(dest::AbstractArray{UInt8,3}, rb::ReplayBuffer, end_idx::Int)
    # dest is (H,W,nf) view or array
    H, W, nf = size(dest)
    @assert nf == rb.n_frames
    start = end_idx - nf + 1

    if start >= 1 || rb.size < rb.capacity
        # contiguous slice, or buffer not yet full
        @inbounds @views dest .= rb.frames[:,:,start:end_idx]
    else
        # wrap-around (only when capacity reached)
        shift = 1 - start                     # number taken from tail
        tail_first = rb.capacity - (shift - 1)
        tail_last  = rb.capacity
        head_last  = end_idx
        @inbounds @views begin
            dest[:,:,1:shift]      .= rb.frames[:,:,tail_first:tail_last]
            dest[:,:,(shift+1):nf] .= rb.frames[:,:,1:head_last]
        end
    end
    return dest
end

# ------------ Allocating sampler (packed, type-stable) ------------
"""
sample_batch(rb, B; rng) -> named tuple:
  states      :: UInt8[H,W,nf,B]
  next_states :: UInt8[H,W,nf,B]  (zeros where terminal)
  next_mask   :: BitVector(B)     (true if non-terminal)
  actions     :: Vector{Int16}
  rewards     :: Vector{Float32}
  dones       :: BitVector
  idxs        :: Vector{Int}
"""
function sample_batch(rb::ReplayBuffer, batch_size::Int; rng=Random.default_rng())
    @assert rb.size >= rb.n_frames "Not enough frames to sample yet."
    H, W, nf = rb.H, rb.W, rb.n_frames

    idxs       = Vector{Int}(undef, batch_size)
    actions    = Vector{Int16}(undef, batch_size)
    rewards    = Vector{Float32}(undef, batch_size)
    dones      = BitVector(undef, batch_size)
    next_mask  = trues(batch_size)

    states      = Array{UInt8}(undef, H, W, nf, batch_size)
    next_states = Array{UInt8}(undef, H, W, nf, batch_size)
    fill!(next_states, 0x00)

    @inbounds for j in 1:batch_size
        i = sample_index(rb, rng)
        idxs[j]    = i
        actions[j] = rb.actions[i]
        rewards[j] = rb.rewards[i]
        d          = rb.dones[i]; dones[j] = d

        copy_stack!(view(states, :,:,:,j), rb, i)

        if !d
            ip1 = (i == rb.capacity ? 1 : i + 1)
            copy_stack!(view(next_states, :,:,:,j), rb, ip1)
        else
            next_mask[j] = false
        end
    end

    return (idxs=idxs, states=states, next_states=next_states,
            next_mask=next_mask, actions=actions, rewards=rewards, dones=dones)
end

# ------------ Zero-allocation sampler (write into preallocated buffers) ------------
"""
sample_batch!(rb, states, next_states, next_mask, actions, rewards, dones, idxs; rng)

Writes in-place to the provided arrays/vectors. Shapes required:
  states/next_states :: UInt8[H,W,nf,B]
  next_mask          :: BitVector(B)
  actions            :: Vector{Int16}(B)
  rewards            :: Vector{Float32}(B)
  dones              :: BitVector(B)
  idxs               :: Vector{Int}(B)
"""
function sample_batch!(rb::ReplayBuffer,
                       states::AbstractArray{UInt8,4},
                       next_states::AbstractArray{UInt8,4},
                       next_mask::BitVector,
                       actions::Vector{Int16},
                       rewards::Vector{Float32},
                       dones::BitVector,
                       idxs::Vector{Int};
                       rng=Random.default_rng())
    @assert rb.size >= rb.n_frames "Not enough frames to sample yet."
    H, W, nf, B = size(states)
    @assert (H, W, nf) == (rb.H, rb.W, rb.n_frames)
    @assert size(next_states) == (H, W, nf, B)
    @assert length(next_mask) == B == length(actions) == length(rewards) == length(dones) == length(idxs)

    fill!(next_states, 0x00)

    @inbounds for j in 1:B
        i = sample_index(rb, rng)
        idxs[j]    = i
        actions[j] = rb.actions[i]
        rewards[j] = rb.rewards[i]
        d          = rb.dones[i]; dones[j] = d

        copy_stack!(view(states, :,:,:,j), rb, i)

        if !d
            next_mask[j] = true
            ip1 = (i == rb.capacity ? 1 : i + 1)
            copy_stack!(view(next_states, :,:,:,j), rb, ip1)
        else
            next_mask[j] = false
        end
    end
    return nothing
end

end # module
