using DifferentialEquations
using Flux
using Plots

k = 1.0
force(dx,x,k,t) = -k*x + 0.1*sin(x)
prob = SecondOrderODEProblem(force,1.0,0.0,(0.0,10.0),k)
sol = solve(prob)

#plot(sol, label=["Velocity", "Position"])

plot_t = 0:0.01:10
data_plot = sol(plot_t)
positions_plot = [state[2] for state in data_plot]
force_plot = [force(state[1], state[2], k, plot_t) for state in data_plot]

# Generate the dataset
t = 0:3.3:10
dataset = sol(t)
position_data = [state[2] for state in sol(t)]
force_data = [force(state[1], state[2], k, t) for state in dataset]

plot(plot_t, force_plot, xlabel="t", label="True Force")
scatter!(t, force_data, label="Foce Measurements")

NNForce = Chain(x -> [x], Dense(1,32,tanh), Dense(32,1), first)

loss() = sum(abs2, NNForce(position_data[i]) - force_data[i] for i in 1:length(position_data))

# Training
opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function() # callback function to observe training
    global iter += 1
    if iter % 500 == 0
        display(loss())
    end
end
display(loss())

random_positions = [2rand()-1 for i in 1:100]
loss_ode() = sum(abs2, NNForce(x) - (-k*x) for x in random_positions)
loss_ode()

λ = 0.1
composed_loss() = loss() + λ*loss_ode()

Flux.train!(composed_loss, Flux.params(NNForce), data, opt; cb=cb)

learned_force_plot = NNForce.(positions_plot)

plot(plot_t, force_plot, xlabel="t", label="True Force")
plot!(plot_t, learned_force_plot, label="Predicted Force")
scatter!(t, force_data, label="Force Measurements")
