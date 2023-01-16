## ------------------------- ##
##      Pendulum Solver      ##
## ------------------------- ##
using PrettyTables
using Plots, Printf
using DelimitedFiles
using Flux
using Statistics
using LinearAlgebra
using ProgressBars

mutable struct Data
    phi::Vector
    time::Vector
    Data() = new()
end

show_plot = false
save_plot = false

## File Loaded and Case Selection
println("-- Pendulum Solver --")
case = "central_diff"
#case = "euler"


## Load Pendulum Object
include("Dynsys.jl")
intitial_phi = 1.0
intitial_vel = 0.0

l = 1.0 / 30.0
g = 10.0
m = 1.0
c = 5.0
k = 0.0
pendulum = Dynsys.Math_pendulum(l, g, m, c, k, intitial_phi, intitial_vel)


## Load Integrator Object
ts = 2000
dt = 1.0 / ts
Integ = Dynsys.Integrator(dt, ts)

data = Data()
data.phi = zeros(ts)
data.time = zeros(ts)


## Setup Time Intetration
# initial setting
if show_plot == true
    fig = Dynsys.create_fig(pendulum)
    Dynsys.plot_state(pendulum)
    display(fig)
end

# compute phi_-1
if case == "central_diff"
    acceleration = -pendulum.g / pendulum.l * sin(pendulum.phi)
    pendulum.phi_prev = pendulum.phi + 0.5 * Integ.delta_t * Integ.delta_t * acceleration
else
    pendulum.phi_prev = 0
end



## ------------------------- ##
##      Data Deneration      ##
## ------------------------- ##
println("Using '" * case * "' Solver!\nGenerating Data")
for i in 1:Integ.timesteps
    # integration step
    Dynsys.run_step(Integ, case, pendulum)

    # plot the state
    if show_plot == true
        fig = Dynsys.create_fig(pendulum)
        Dynsys.plot_state(pendulum)
        display(fig)
    end

    # save the step
    data.phi[i] = pendulum.phi
    data.time[i] = dt * i
end
println("Data Generation Complete")

if save_plot == true
    println("Saving Data Plot")
    plot(data.time,data.phi)
    savefig(case * ".png")
    println("Save Complete")
end

plot(data.time,data.phi)
## ------------------------- ##
##       Neural Network      ##
## ------------------------- ##
## Network Data
training_ts = Int(floor(0.35* ts))
training_data_t = data.time[1:training_ts]'  # time values
training_data_y = data.phi[1:training_ts]'
plot(training_data_t',training_data_y')
## Network Creation
neural_network = Chain(
    Dense(1, 40, tanh),
    Dense(40,40, tanh),
    Dense(40, 1)
)



function loss_function(x, y)
    predict = neural_network(x)     # Get predicted phi values
    mse = mean((y .- predict).^2)   # Compute mse loss

    physics = neural_network(collect(range(0,1,step=0.001))')                           # Get some phi values for the physics loss
    v = (physics[3:end] .- physics[1:end-2]) / (2*0.001)                                # Compute phi_dot using central difference
    a = (physics[3:end] .- 2*physics[2:end-1] .+ physics[1:end-2]) / (0.001*0.001)      # Compute phi_ddot using central difference
    res = (m * a) .+ (c * v) .+ (g / l )*(physics[2:end-1])                            # Compute the physics residual

    loss = mse + 1e-4* mean(abs2.(res))  # Compute the total loss function
    #println(loss)                           # Print loss to see if converging
    return loss                             # Return loss
end

optimizer = Adam(0.01)                                                                              # Selected optimizer
parameters = Flux.params(neural_network)                                                            # Get NN parameters
input_data = Flux.Data.DataLoader((training_data_t, training_data_y), batchsize=50, shuffle=true)   # Load data
#input_data = [(training_data_t, training_data_y)]

tolerance = 1e-6  # Early termination tolerance
max_itrs = 1000   # Maximum number of training iterations/



## Main Training Loop
println("Training Network")
for e in ProgressBar(1:max_itrs)
    Flux.train!(loss_function, parameters, input_data, optimizer)   # Training call
end
println("Training Complete")


## Plot Trained Network Output and Compare
trained_y = neural_network(data.time')
plot(data.time, [trained_y', data.phi])
plot(data.time, trained_y')
println("Saving Fitted Plot")
savefig("FittedData.png")
println("Save Complete")
