using Plots, CUDA, Random, Enzyme

dt = Float32(0.01)
N = 1000
M = 10
k = 10
epsilon = 1e-6

function step!(
    pos_before :: CuArray{Float32, 2, CUDA.Mem.DeviceBuffer},
    pos_after :: CuArray{Float32, 2, CUDA.Mem.DeviceBuffer},
    vel_before :: CuArray{Float32, 2, CUDA.Mem.DeviceBuffer},
    vel_after :: CuArray{Float32, 2, CUDA.Mem.DeviceBuffer},
    attr_pos :: CuArray{Float32, 2, CUDA.Mem.DeviceBuffer},
    attr_s :: CuArray{Float32, 1, CUDA.Mem.DeviceBuffer},
    dt :: Float32)

    diff_x = pos_before[1, :] .- attr_pos[1, :]'
    diff_y = pos_before[2, :] .- attr_pos[2, :]'

    distance_sq = (diff_x.^2 .+ diff_y.^2) .+ epsilon
    
    forces = attr_s' ./ distance_sq

    forces_x = forces .* diff_x
    forces_y = forces .* diff_y 

    vel_after[1, :] = vel_before .- sum(forces_x, dims=2) * dt
    vel_after[2, :] = vel_before .- sum(forces_y, dims=2) * dt

    pos_after[1, :] .+= vel_after[1, :] .* dt
    pos_after[2, :] .+= vel_after[2, :] .* dt

    return nothing
end

pos_0_h = zeros(Float32, (k, 2, N))
pos_0_h[1, :, :] .= Random.rand(Float32, (2, N)) .- 0.5

vel_0_h = zeros(Float32, (k, 2, N))
attr_pos_h = Random.rand(Float32, (2, M)) .- 0.5
attr_s_h = Random.rand(Float32, M)

pos_d = CuArray{Float32}(pos_0_h)
vel_d = CuArray{Float32}(vel_0_h)
attr_pos_d = CuArray{Float32}(attr_pos_h)
attr_s_d = CuArray{Float32}(attr_s_h)

function forward!(
    pos_hist :: CuArray{Float32, 3, CUDA.Mem.DeviceBuffer},
    vel_hist :: CuArray{Float32, 3, CUDA.Mem.DeviceBuffer},
    attr_pos :: CuArray{Float32, 2, CUDA.Mem.DeviceBuffer},
    attr_s :: CuArray{Float32, 1, CUDA.Mem.DeviceBuffer},
    dt :: Float32,
    steps :: Int)

    for i in 1:(steps-1)
        step!(
            pos_hist[i, :, :], pos_hist[i+1, :, :], 
            vel_hist[i, :, :], vel_hist[i+1, :, :],
            attr_pos, attr_s, dt)
    end

    return nothing
end

function loss(
    pos :: CuArray{Float32, 2, CUDA.Mem.DeviceBuffer},
    pos_star :: CuArray{Float32, 2, Cuda.Mem.DeviceBuffer},
    loss :: CuArray{Float32, 2, Cuda.Mem.DeviceBuffer})

    return sum(pos - pos_star .^2)

function backward!(
    pos_hist :: CuArray{Float32, 3, CUDA.Mem.DeviceBuffer},
    pos_star :: CuArray{Float32, 2, CUDA.Mem.DeviceBuffer},
    adj_pos_before :: CuArray{Float32, 2, CUDA.Mem.DeviceBuffer},
    adj_pos_after :: CuArray{Float32, 2, CUDA.Mem.DeviceBuffer},
    loss : CuArray{Float32, 1, CUDA.Mem.DeviceBuffer},
    vel_hist :: CuArray{Float32, 3, CUDA.Mem.DeviceBuffer},
    attr_pos :: CuArray{Float32, 2, CUDA.Mem.DeviceBuffer},
    attr_s :: CuArray{Float32, 1, CUDA.Mem.DeviceBuffer},
    dt :: Float32,
    steps :: Int)

    loss_value = loss(pos_hist[end, :, :], pos_star, loss)

    autodiff(
        Reverse,
        loss,
        Duplicated(pos_hist[end, :, :], b_pos),

    for i in steps:1
        pos_before = pos_hist[i, :, :]
        pos_after = pos_hist[i+1, :, :]

        autodiff(
            Reverse,
            step,
            Duplicated(pos_before, adj_pos_before),
            Duplicated(pos_after, adj_pos_after),
            Const(vel_hist[i, :, :]),
            Const(vel_hist[i+1, :, :]),
            Const(attr_pos),
            Const(attr_s),
            Const(dt))
    end
end

# forward pass
forward!(
    pos_d,
    vel_d,
    attr_pos_d,
    attr_s_d,
    dt,
    steps)
 
# backward pass
backward!(
    pos_d,
    vel_d,
    attr_pos_d,
    attr_s_d,
    dt,
    steps)

# transfer particles positions to host
pos_h = Array(pos_d)

p1 = scatter(
    pos_0_h[1, :], pos_0_h[2, :],
    label="Particle", markersize=2, 
    markerstrokewidth=0, xlims=(-1, 1), ylims=(-1, 1))

p1 = scatter!(
    attr_pos_h[1, :], attr_pos_h[2, :],
    label="Attractor", markersize=5, markerstrokewidth=0)

p2 = scatter(
    pos_h[1, :], pos_h[2, :],
    label="Particle", markersize=2, 
    markerstrokewidth=0, xlims=(-1, 1), ylims=(-1, 1))

p2 = scatter!(
    attr_pos_h[1, :], attr_pos_h[2, :],
    label="Attractor", markersize=5, markerstrokewidth=0)

plt = plot(p1, p2, layout=(1, 2), size=(1200, 600))

# limit 

gui(plt)

savefig(plt, "particles.png")