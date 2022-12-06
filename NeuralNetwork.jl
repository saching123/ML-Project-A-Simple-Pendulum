using Flux
using Plots
using DataFrames
using CSV
using Flux: params
using Flux: train!
dataset = Matrix(DataFrame(CSV.File("datalearning1.csv")))
xs = dataset[:,1]
ys = dataset[:,2]            
layer1 = Dense(1,64,tanh)
layer2 = Dense(64,64,gelu)
layer3 = Dense(64,64,leakyrelu)
layer4 = Dense(64,1,tanh)
#layer3 = Dense(3,1,σ)
model  = Chain(layer1, layer2,layer3,layer4)
loss(x,y) = Flux.mse(model(x),y)
params(model)
data = [(xs',ys')]
opt = Adam(0.001)
loss(xs',ys')
model(xs')
ys'
count = 0
for i in 1:4000
    train!(loss, params(model), data, opt)
    plot(xs,model(xs')')
end
     count+=10
  
     model(xs') 
  plot(xs,model(xs')')

     ys'
dataset = Matrix(DataFrame(CSV.File("datavalidation1.csv")))
xs1 = dataset[:,1]
ys1 = dataset[:,2]            
model(xs1')
ys'

loss(xs1', ys1')
plot(xs1,[model(xs1')', ys1 ])
plot(xs, ys)
model(xs')
ys