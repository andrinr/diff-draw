using Plots, CUDA, Enzyme

dt = Float32(0.01)
N = 10^3
M = 10
steps = 20
epsilon = 10^-5

struct Particle
    pos_x::CuArray{Float32, 1}
    pos_y::CuArray{Float32, 1}
    vel_x::CuArray{Float32, 1}
    vel_y::CuArray{Float32, 1}
    mass::CuArray{Float32, 1}
end

println("Initializing particles...")

particles = Particle(
    CUDA.rand(Float32, N),
    CUDA.rand(Float32, N),
    CUDA.rand(Float32, N),
    CUDA.rand(Float32, N),
    CUDA.rand(Float32, N)
)

println("Particles initialized!")

function step!(particles :: Particle, dt :: Float32)
    diff_x = particles.pos_x .- particles.pos_x'
    diff_y = particles.pos_y .- particles.pos_y'

    distance = (diff_x.^2 .+ diff_y.^2) .+ epsilon
    
    forces = (particles.mass .* particles.mass') ./ distance

    distance = sqrt.(distance)

    forces_x = (forces .* diff_x) ./ distance
    forces_y = (forces .* diff_y) ./ distance

    particles.vel_x .-= sum(forces_x, dims=2) .* dt
    particles.vel_y .-= sum(forces_y, dims=2) .* dt

    particles.pos_x .+= particles.vel_x .* dt
    particles.pos_y .+= particles.vel_y .* dt

end

function simulate(particles :: Particle, steps :: Int, dt :: Float32)
    for i in 1:steps
        step!(particles, dt)
    end
end

println("Running simulation...")

simulate(particles, steps, dt)

println("Done!")

# transfer particles positions to host
pos_x_h = Array(particles.pos_x)
pos_y_h = Array(particles.pos_y)

# dark theme
theme(:dark)

# scatter with no stroke / borders
plt = scatter(
    pos_x_h, pos_y_h, 
    label="Particles", markersize=1, 
    markercolor=:white, markerstrokewidth=0)

gui(plt)

savefig(plt, "imgs/particles.png")