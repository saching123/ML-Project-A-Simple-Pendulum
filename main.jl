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

length = 1.0/30.0
gravity = 10.0
mass = 1.0
damping = 5.0
phi = 1.0
phi_dot = 0.0
delta_t = 0.001
timesteps = 2000

pendulum = Dynsys.Math_pendulum(length, gravity, mass, damping, phi, phi_dot)
## load integrator and memory for the results
Integ = Dynsys.Integrator(delta_t,timesteps)
Integ.res_phi = zeros(Integ.timesteps)
Integ.res_phi_dot = zeros(Integ.timesteps)
initial_angle = pendulum.phi
initial_angular_velocity = pendulum.phi_dot


## run time integration
# initial setting
fig = Dynsys.create_fig(pendulum)
Dynsys.plot_state(pendulum)
display(fig)

# running over the time step
for i in 1:Integ.timesteps
    fig = Dynsys.create_fig(pendulum)
    Dynsys.run_step(Integ, "euler", pendulum)
    #Dynsys.plot_state(pendulum)
    #display(fig)
    # save the step
    Integ.res_phi[i] = pendulum.phi
    Integ.res_phi_dot[i] = pendulum.phi_dot
end

x = range(1, Integ.timesteps)

# inserting initial values of phi and phi_dot before plotting
insert!(Integ.res_phi, 1, initial_angle)
insert!(Integ.res_phi_dot, 1, initial_angular_velocity)

## normalizing x and y axis values here
x_normalized = [(i/Integ.timesteps) for i in 1:Integ.timesteps]
y_normalized = [(Integ.res_phi[i]/Integ.res_phi[1]) for i in 1:Integ.timesteps]

## plot the normalized values
plot(x_normalized, y_normalized, color="red", lw=2.0)


## store data in a csv file
df1 = DataFrame(A = x_normalized, B = y_normalized)
CSV.write("datalearning1.csv", df1, append = true)
