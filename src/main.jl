include("DataModule.jl")
include("RecurrentNetworkModule.jl")
include("DenseNetworkModule.jl")
include("AccuracyModule.jl")
include("GradientOptimizersModule.jl")

using .DataModule, .RecurrentNetworkModule, .DenseNetworkModule, .AccuracyModule, .GradientOptimizersModule
using Plots

struct LayerWrapper
    layer
    optimizer::GradientOptimizer
end

function forward(model, input::Matrix{Float32})
    for layer in model
        input = layer.layer(input)
    end
    return input
end

function train(model, data::Matrix{Float32}, data_y)
    result = forward(model, data)
    loss, acc, C = AccuracyModule.loss_and_accuracy(result, data_y)

    for layer in reverse(model)
        C = layer.optimizer(C)
        if isa(layer.layer, DenseNetworkModule.Dense)
            C = DenseNetworkModule.back(layer.layer, C)
        end
        if isa(layer.layer, RecurrentNetworkModule.RNNCell)
            C = RecurrentNetworkModule.back(layer.layer, C)
        end
    end
    return result
end

function main()
    # (in) => (out)
    rnn = LayerWrapper(RecurrentNetworkModule.RNN(784 => 64, tanh), Descent(1f-10))
    dense1 = LayerWrapper(DenseNetworkModule.Dense(784 => 64, identity), Descent(0.00000000002f0))
    dense2 = LayerWrapper(DenseNetworkModule.Dense(64 => 10, identity), Descent(1f0))
    model = [rnn, dense2]
    batch_size = 100
    epochs = 5

    println("Loading training data...")
    train_x, train_y = DataModule.preprocess(:train; one_hot = true)

    train_x_batched = DataModule.batch(train_x, batch_size)
    train_y_batched = DataModule.batch(train_y, batch_size)

    println("Loading test data...")
    test_x, test_y = DataModule.preprocess(:test; one_hot = true)

    println("Training...")
    batch_acc = Float64[]
    for epoch in 1:epochs
        batches = size(train_x_batched, 1)
        @time begin
            for batch in 1:batches
                result = train(model, train_x_batched[batch], train_y_batched[batch])
                loss, acc, _ = AccuracyModule.loss_and_accuracy(result, train_y_batched[batch])
                push!(batch_acc, acc)
            end
        end
        result = forward(model, train_x)
        _, train_acc, _ = AccuracyModule.loss_and_accuracy(result, train_y)
        result = forward(model, test_x)
        _, test_acc, _ = AccuracyModule.loss_and_accuracy(result, test_y)
        @info epoch train_acc test_acc
    end

    plot(batch_acc, xlabel="Batch num", ylabel="acc", title="Accuracy over batches")
end
