module Particles

using Plots, CUDA, Random, Enzyme

dt = Float32(0.01)
N = 10000
M = 10
epsilon = 1e-6
steps = 100

function step!(
    pos_before :: AbstractArray{Float32, 2},
    pos_after :: AbstractArray{Float32, 2},
    vel_before :: AbstractArray{Float32, 2},
    vel_after :: AbstractArray{Float32, 2},
    attr_pos :: AbstractArray{Float32, 2},
    attr_s :: AbstractArray{Float32, 1},
    dt :: Float32) :: Nothing

    diff_x = pos_before[1, :] .- attr_pos[ 1, :]'
    diff_y = pos_before[2, :] .- attr_pos[ 2, :]'

    distance_sq = (diff_x.^2 .+ diff_y.^2) .+ epsilon
    
    forces = attr_s' ./ distance_sq

    forces_x = forces .* diff_x
    forces_y = forces .* diff_y 

    vel_after[1, :] = vel_before[1, :] .- sum(forces_x, dims=2) * dt
    vel_after[2, :] = vel_before[2, :] .- sum(forces_y, dims=2) * dt

    pos_after[1, :] = pos_before[1, :] .+ vel_after[1, :] .* dt
    pos_after[2, :] = pos_before[2, :] .+ vel_after[2, :] .* dt

    return nothing

end

println("Initializing...")
pos_0_h = zeros(Float32, (steps, 2, N))
pos_0_h[1, :, :] = Random.rand(Float32, (2, N)) .- 0.5

vel_0_h = zeros(Float32, (steps, 2, N))
attr_pos_h = Random.rand(Float32, (2, M)) .- 0.5
attr_s_h = Random.rand(Float32, M)

pos_d = CuArray{Float32}(pos_0_h)
vel_d = CuArray{Float32}(vel_0_h)
attr_pos_d = CuArray{Float32}(attr_pos_h)
attr_s_d = CuArray{Float32}(attr_s_h)

function forward!(
    pos_hist :: AbstractArray{Float32, 3},
    vel_hist :: AbstractArray{Float32, 3},
    attr_pos :: AbstractArray{Float32, 2},
    attr_s :: AbstractArray{Float32, 1},
    dt :: Float32,
    steps :: Int) :: Nothing

    for i in 1:(steps-1)

        println("Step: ", i)

        step!(
            view(pos_hist, i, :, :), view(pos_hist, i+1, :, :), 
            view(vel_hist, i, :, :), view(vel_hist, i+1, :, :),
            attr_pos, attr_s, dt)

    end

    return nothing
end

function backward!(
    pos_hist :: AbstractArray{Float32, 3},
    vel_hist :: AbstractArray{Float32, 3},
    attr_pos :: AbstractArray{Float32, 2},
    attr_s :: AbstractArray{Float32, 1},
    dt :: Float32,
    steps :: Int) :: Nothing

    zero_h = zeros(Float32, (2, N))
    adj_ping_pong_h = zeros(Float32, (2, 2, N))
    adj_ping_pong_h[1, 1, 1] = 1

    zero = CuArray{Float32}(zero_h)
    adj_ping_pong = CuArray{Float32}(adj_ping_pong_h)

    for i in steps:-1:1

        pos_before = view(pos_hist, i, :, :)
        vel_before = view(vel_hist, i, :, :)

        view(adj_ping_pong, (i+1) % 2, :, :) .= 0
        pos_after_adj = view(adj_ping_pong, i % 2, :, :)
        pos_before_adj = view(adj_ping_pong, (i+1) % 2, :, :)

        pos_after_adj .= 0

        autodiff(
            Reverse,
            step,
            Duplicated(pos_before, pos_before_adj)),
            Duplicated(zero, pos_after_adj),
            Const(vel_before),
            Const(zero),
            Const(attr_pos),    
            Const(attr_s),
            Const(dt))

        if i == 1
            return view(adj_ping_pong, i % 2, :, :)
        end

    end

end

println("Forward pass...")
# profile = CUDA.@profile
forward!(
    pos_d,
    vel_d,
    attr_pos_d,
    attr_s_d,
    dt,
    steps)

println("Backward pass...")
backward!(
    pos_d,
    vel_d,
    attr_pos_d,
    attr_s_d,
    dt,
    steps)

# println(profile)

pos_h = Array(pos_d)

println("Plotting...")
p1 = scatter(
    pos_0_h[1, 1, :], pos_0_h[1, 2, :],
    label="Particle", markersize=2, 
    markerstrokewidth=0, xlims=(-1, 1), ylims=(-1, 1))

p1 = scatter!(
    attr_pos_h[1, :], attr_pos_h[2, :],
    label="Attractor", markersize=5, markerstrokewidth=0)

p2 = scatter(
    pos_h[end, 1, :], pos_h[end, 2, :],
    label="Particle", markersize=2, 
    markerstrokewidth=0, xlims=(-1, 1), ylims=(-1, 1))

p2 = scatter!(
    attr_pos_h[1, :], attr_pos_h[2, :],
    label="Attractor", markersize=5, markerstrokewidth=0)

plt = plot(p1, p2, layout=(1, 2), size=(1200, 600))

# limit 

gui(plt)

savefig(plt, "particles.png")

end