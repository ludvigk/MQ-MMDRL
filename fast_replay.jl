module FastReplay

using Random

"""
A compact replay buffer storing raw `UInt8` frames.

- frames::Array{UInt8,3} for grayscale:  (H, W, capacity)
  (for RGB use Array{UInt8,4}: (3, H, W, capacity); trivial to extend)
- actions::Vector{UInt8 or Int16}
- rewards::Vector{Float32}
- dones::BitVector
- episode_id::Vector{Int32}     (monotone, increases on reset)
- fsr::Vector{Int32}            frames_since_reset at index i (O(1) validity check)

We assume n=1 bootstrapping for speed; if `done[i]`, we do not construct next_state.
"""
mutable struct ReplayBuffer
    H::Int
    W::Int
    n_frames::Int
    capacity::Int

    frames::Array{UInt8,3}   # (H, W, capacity)
    actions::Vector{Int16}
    rewards::Vector{Float32}
    dones::BitVector

    episode_id::Vector{Int32}
    fsr::Vector{Int32}       # frames_since_reset

    size::Int                # how many entries filled (<= capacity)
    write_pos::Int           # 1-based ring index
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

@inline function _incpos(pos::Int, cap::Int)
    np = pos + 1
    np > cap ? 1 : np
end

"""
Push a transition:
- `frame`    : UInt8, (H, W)
- `action`   : Int
- `reward`   : Real
- `done`     : Bool
"""
function Base.push!(rb::ReplayBuffer, frame::AbstractArray{UInt8,2}, action, reward, done::Bool)
    @inbounds rb.frames[:,:,rb.write_pos] = frame
    @inbounds rb.actions[rb.write_pos] = Int16(action)
    @inbounds rb.rewards[rb.write_pos] = Float32(reward)
    @inbounds rb.dones[rb.write_pos] = done

    # episode id and frames_since_reset in O(1)
    if rb.write_pos == 1 && rb.size == rb.capacity
        # wrapping overwrite; derive previous pos
        prev = rb.capacity
    else
        prev = (rb.write_pos == 1) ? max(rb.size, 1) : rb.write_pos - 1
    end

    if rb.size == 0
        rb.episode_id[rb.write_pos] = rb.current_episode
        rb.fsr[rb.write_pos] = 1
    else
        rb.episode_id[rb.write_pos] = rb.current_episode
        rb.fsr[rb.write_pos] = rb.dones[prev] ? 1 : rb.fsr[prev] + 1
    end

    # advance pointers
    if done
        rb.current_episode += 1
    end

    if rb.size < rb.capacity
        rb.size += 1
    end
    rb.write_pos = _incpos(rb.write_pos, rb.capacity)
    rb
end

"""
Quick check: is index `i` valid to form a stacked state of length `rb.n_frames`?
O(1): we just check frames_since_reset at i.
"""
@inline valid_state_at(rb::ReplayBuffer, i::Int) = (rb.fsr[i] >= rb.n_frames)

"""
Uniformly sample an index with a valid stacked state.
Rejection sampling is OK because after warmup most indices are valid.
"""
function sample_index(rb::ReplayBuffer, rng::AbstractRNG)
    @assert rb.size >= rb.n_frames "Not enough frames to sample yet."
    lo = (rb.size == rb.capacity) ? 1 : 1          # full range [1, rb.size]
    hi = rb.size
    while true
        i = rand(rng, lo:hi)
        if valid_state_at(rb, i)
            return i
        end
    end
end

"""
Assemble stacked state ending at index `i`.
Returns UInt8 array of shape (H, W, n_frames) with frames [i-n_frames+1 ... i].
"""
function stacked_state(rb::ReplayBuffer, i::Int)
    @assert valid_state_at(rb, i)
    H, W, nf = rb.H, rb.W, rb.n_frames
    s = Array{UInt8}(undef, H, W, nf)
    # copy slices in order (oldest to newest)
    start = i - nf + 1
    # Handle wrap-around only if buffer is at capacity and start < 1
    if start >= 1 || rb.size < rb.capacity
        @inbounds for k in 1:nf
            s[:,:,k] = rb.frames[:,:,start + k - 1]
        end
    else
        # wrap: split copy from [i-nf+1 .. 0] and [1 .. i]
        # convert negative to tail indices
        shift = 1 - start
        # tail part
        @inbounds for k in 1:shift
            s[:,:,k] = rb.frames[:,:,rb.capacity - (shift - k)]
        end
        # head part
        @inbounds for k in (shift+1):nf
            s[:,:,k] = rb.frames[:,:,k - shift]
        end
    end
    s
end

@inline function copy_stack!(dest::AbstractArray{UInt8,3}, rb, end_idx::Int)
    # dest is (H,W,nf) and may be a SubArray (from view(states, :,:,:,j))
    H, W, nf = size(dest)
    @assert nf == rb.n_frames
    start = end_idx - nf + 1

    if start >= 1 || rb.size < rb.capacity
        # no wrap, or buffer not full → contiguous slice
        @inbounds @views dest .= rb.frames[:,:,start:end_idx]
    else
        # wrap-around (only when buffer is full)
        # number of frames we need from the tail end of the ring
        shift = 1 - start                      # == -start + 1
        tail_first = rb.capacity - (shift - 1) # inclusive
        tail_last  = rb.capacity               # inclusive
        head_last  = end_idx                   # inclusive

        @inbounds @views begin
            # tail part fills the first `shift` frames
            dest[:,:,1:shift] .= rb.frames[:,:,tail_first:tail_last]
            # head part fills the remaining frames
            dest[:,:,(shift+1):nf] .= rb.frames[:,:,1:head_last]
        end
    end
    return dest
end

"""
sample_batch(rb, B; rng) -> named tuple with packed arrays:
  states      :: UInt8[H,W,nf,B]
  next_states :: UInt8[H,W,nf,B]  (zeros for terminals)
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
    fill!(next_states, 0x00)  # terminals remain zero

    @inbounds for j in 1:batch_size
        i = sample_index(rb, rng)
        idxs[j]    = i
        actions[j] = rb.actions[i]
        rewards[j] = rb.rewards[i]
        d          = rb.dones[i]; dones[j] = d

        # state stack ending at i
        copy_stack!(view(states, :,:,:,j), rb, i)

        # next_state only if non-terminal
        if !d
            ip1 = (i == rb.capacity) ? 1 : i + 1
            copy_stack!(view(next_states, :,:,:,j), rb, ip1)
        else
            next_mask[j] = false
        end
    end

    return (idxs=idxs, states=states, next_states=next_states,
            next_mask=next_mask, actions=actions, rewards=rewards, dones=dones)
end

end # module
