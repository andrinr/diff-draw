using Plots, CUDA, Distances

dt = Float32(0.01)
N = 100000
M = 10
steps = 100

struct Particle
    pos_x::CuArray{Float32, 1}
    pos_y::CuArray{Float32, 1}
    vel_x::CuArray{Float32, 1}
    vel_y::CuArray{Float32, 1}
end

struct Attractor
    pos_x::CuArray{Float32, 1}
    pos_y::CuArray{Float32, 1}
    strength::CuArray{Float32, 1}
end

particles = Particle(
    CUDA.rand(Float32, N),
    CUDA.rand(Float32, N),
    CUDA.rand(Float32, N),
    CUDA.rand(Float32, N)
)

attractors = Attractor(
    CUDA.rand(Float32, M),
    CUDA.rand(Float32, M),
    CUDA.rand(Float32, M)
)

function step!(particles :: Particle, attractors :: Attractor, dt :: Float32)
    diff_x = particles.pos_x .- attractors.pos_x'
    diff_y = particles.pos_y .- attractors.pos_y'

    distance_sq = (diff_x.^2 .+ diff_y.^2)
    
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
pos_x_h = Array(particles.pos_x)
pos_y_h = Array(particles.pos_y)
attr_pos_x_h = Array(attractors.pos_x)
attr_pos_y_h = Array(attractors.pos_y)

# scatter with no stroke / borders
plt = scatter(
    pos_x_h, pos_y_h, 
    label="Particles", markersize=2, 
    markercolor=:blue, markerstrokewidth=0)

scatter!(
    attr_pos_x_h, attr_pos_y_h, 
    label="Attractors", markersize=2, 
    markercolor=:red, markerstrokewidth=0)

gui(plt)

savefig(plt, "particles.png")

