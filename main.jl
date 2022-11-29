## running Euler integrator for the mathematical pendulum
######################################
######################################
######################################

#Juno.clearconsole()
## header files 

using PrettyTables
using Plots, Printf
using DelimitedFiles
using DataFrames, CSV

## booleans

show_video = false

## starting the file
println("-- pendulum euler --")

## load pendulum

include("Dynsys.jl")

length = 1.0
gravity = 10.0
mass = 1.0
damping = 1.0
stiffness = 300.0
phi = 0.5
phidot = 0.0
delta_t = 0.02
timesteps = 400

pendulum = Dynsys.Math_pendulum(length, gravity, mass, damping, stiffness, phi, phidot)
## load integrator and memory for the results

Integ = Dynsys.Integrator(delta_t,timesteps)
Integ.res_phi = zeros(Integ.timesteps)
Integ.res_phi_dot = zeros(Integ.timesteps)

## run time integration
# initial setting

fig = Dynsys.create_fig(pendulum)
Dynsys.plot_state(pendulum)
display(fig)

# running over the time step

for i in 1:Integ.timesteps
    fig = Dynsys.create_fig(pendulum)
    Dynsys.run_step(Integ, "euler", pendulum)
    Dynsys.plot_state(pendulum)
    display(fig)
    # save the step
    Integ.res_phi[i] = pendulum.phi
    Integ.res_phi_dot[i] = pendulum.phi_dot
end

x = range(1, Integ.timesteps)

## normalizing x and y axis values here

x_normalized = [(i/Integ.timesteps) for i in 1:Integ.timesteps]
norm = [(Integ.res_phi[i]/Integ.res_phi[1]) for i in 1:Integ.timesteps]

## plot the normalized values

plot(x_normalized, norm, color="red", lw=2.0)

## store data in a tabular format
df2 = DataFrame(Length = "l", Gravity = "g", Mass = "m", Damping = "c", ϕ = "phi", ϕ_dot = "phi_dot", Δt = "delta_t", ts = "ts")
df1 = DataFrame(Length = length, Gravity = gravity, Mass = mass, Damping = damping, ϕ = phi, ϕ_dot = phidot, Δt = delta_t, ts = timesteps)
df = DataFrame(A = x_normalized[1:Integ.timesteps], B = norm[1:Integ.timesteps])

## write the values onto CSV file
CSV.write("dataset.csv",df2,append = true)
CSV.write("dataset.csv",df1,append = true)
CSV.write("dataset.csv",df,append = true)
