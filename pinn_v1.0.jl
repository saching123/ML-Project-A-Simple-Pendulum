include("main.jl")
#import necessary packages
using Flux
using Flux: train!
using Plots
using DataFrames, CSV


x_train = [x_normalized[i] for i in 1:1200]
y_train = [y_normalized_phi[i] for i in 1:1200]


# Reshape the data
function reshape_data(x_train, y_train)
    x_train_reshaped = reduce(hcat, x_train)
    y_train_reshaped = reduce(hcat, y_train)
    return x_train_reshaped, y_train_reshaped
end

x_train_reshaped, y_train_reshaped = reshape_data(x_train, y_train)
x_normalized_reshaped, y_normalized_reshaped = reshape_data(x_normalized, y_normalized_phi)

# Define the nodel
model = Chain(Dense(1, 32, tanh), Dense(32, 32, tanh), Dense(32, 1))

predict(x) = model(x)

# Collect the parameters W and b
ps = Flux.params(model)

# Define the loss function
function total_loss(x, y)
    loss_data_driven = Flux.Losses.mse(model(x), y)
    pred_1_phi = predict(x_train_reshaped)
    NN_phi = vec(pred_1_phi)
    NN_phi_dot = [(NN_phi[i+1] - NN_phi[i]) / delta_t for i in 1:1200-1]
    NN_phi_double_dot = [(NN_phi_dot[i+1] - NN_phi_dot[i]) / delta_t for i in 1:1200-2]
    sin_phi = [sin(NN_phi[i]) for i in 1:1200-2]
    residue = (mass*NN_phi_double_dot) + (damping*NN_phi_dot[1:1200-2]) + ((gravity/len)*sin_phi)
    loss_physical = sum(abs2, residue)  / (1200 - 2)
    loss_total = loss_data_driven + 0.00001*loss_physical

    return loss_total
end

# Set an optimization routine
# Use gradient descent
learning_rate = 0.008
opt = Adam(learning_rate)

#pred_0_phi = predict(x_train_reshaped)
#loss_0_phi = loss(predict(x_train_reshaped), y_train_reshaped)

# Zip the train before so we can pass it to the training function
data = [(x_train_reshaped, y_train_reshaped)]

n_epochs = 2000

for epoch in 1:n_epochs
    train!(total_loss, ps, data, opt)
    #println("Epoch: $epoch, loss: ", loss(predict(x_train_reshaped), y_train_reshaped))

end

pred_1_phi = predict(x_normalized_reshaped)

#plot(epochs, loss, label="Loss vs epoch", color="blue", lw=2.0)
plot(x_normalized_reshaped', y_normalized_reshaped', label="Ground truth_phi", color="red", lw=2.0)
plot!(x_normalized_reshaped', pred_1_phi', label="NN_phi", color="green", lw=2.0)

#loss_1 = loss(predict(x_train_reshaped), y_train_reshaped)
#println("Final loss = $loss_1")

# Write to .csv file
#df1 = DataFrame(A = x_normalized, B = vec(pred_1_phi))
#CSV.write("pred_1_phi.csv", df1, append = true)
