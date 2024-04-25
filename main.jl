using Plots, CUDA, Distances

dt = Float32(0.01)
N = 1000
M = 10
steps = 50
epsilon = 1e-6

struct Particle{T}
    pos_x::T
    pos_y::T
    vel_x::T
    vel_y::T
end

struct Attractor{T}
    pos_x::T
    pos_y::T
    strength::T
end

particles = Particle{CuArray{Float32, 1}}(
    CUDA.rand(Float32, N) .- 0.5,
    CUDA.rand(Float32, N) .- 0.5,
    CUDA.rand(Float32, N) .- 0.5,
    CUDA.rand(Float32, N) .- 0.5
)

pos_x_h_0 = Array(particles.pos_x)
pos_y_h_0 = Array(particles.pos_y)

attractors = Attractor{CuArray{Float32, 1}}(
    CUDA.rand(Float32, M) .- 0.5,
    CUDA.rand(Float32, M) .- 0.5,
    CUDA.rand(Float32, M)
)

function step!(particles :: Particle, attractors :: Attractor, dt :: Float32)
    diff_x = particles.pos_x .- attractors.pos_x'
    diff_y = particles.pos_y .- attractors.pos_y'

    distance_sq = (diff_x.^2 .+ diff_y.^2) .+ epsilon
    
    forces = attractors.strength' ./ distance_sq

    forces_x = forces .* diff_x
    forces_y = forces .* diff_y

    particles.vel_x .-= sum(forces_x, dims=2)
    particles.vel_y .-= sum(forces_y, dims=2)

    particles.pos_x .+= particles.vel_x .* dt
    particles.pos_y .+= particles.vel_y .* dt

end

println("Running simulation...")

for i in 1:steps
    step!(particles, attractors, dt)
end

println("Done!")

# transfer particles positions to host
pos_x_h_T = Array(particles.pos_x)
pos_y_h_T = Array(particles.pos_y)

attr_pos_x_h = Array(attractors.pos_x)
attr_pos_y_h = Array(attractors.pos_y)

p1 = scatter(
    pos_x_h_0, pos_y_h_0,
    label="Particle", markersize=2, 
    markerstrokewidth=0, xlims=(-1, 1), ylims=(-1, 1))

p1 = scatter!(
    attr_pos_x_h, attr_pos_y_h,
    label="Attractor", markersize=10, markerstrokewidth=0)

p2 = scatter(
    pos_x_h_T, pos_y_h_T,
    label="Particle", markersize=2, 
    markerstrokewidth=0, xlims=(-5, 5), ylims=(-5, 5))

p2 = scatter!(
    attr_pos_x_h, attr_pos_y_h,
    label="Attractor", markersize=4, markerstrokewidth=0)

plt = plot(p1, p2, layout=(1, 2), size=(1200, 600))

# limit 

gui(plt)

savefig(plt, "particles.png")