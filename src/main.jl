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

function backward(model, C)
    for layer in reverse(model)
        C = layer.optimizer(C)
        if isa(layer.layer, DenseNetworkModule.Dense)
            C = DenseNetworkModule.back(layer.layer, C)
        end
        if isa(layer.layer, RecurrentNetworkModule.RNNCell)
            C = RecurrentNetworkModule.back(layer.layer, C)
        end
    end
end

function forward(model, input::Matrix{Float32})
    for layer in model
        input = layer.layer(input)
    end
    return input
end

function main()
    batch_size = 100
    epochs = 5
    # (in) => (out)
    rnn = LayerWrapper(RecurrentNetworkModule.RNN(196 => 64, tanh), Adagrad(zeros(Float32, 64, batch_size), 10e-8))
    dense = LayerWrapper(DenseNetworkModule.Dense(64 => 10, identity), Descent(10e-1))
    model = [rnn, dense]

    println("Loading training data...")
    train_x, train_y = DataModule.preprocess(:train; one_hot = true)

    train_x_batched = DataModule.batch(train_x, batch_size)
    train_y_batched = DataModule.batch(train_y, batch_size)

    println("Loading test data...")
    test_x, test_y = DataModule.preprocess(:test; one_hot = true)

    println("Training...")
    batch_acc = Float64[]
    batch_loss = Float64[]
    for epoch in 1:epochs
        batches = size(train_x_batched, 1)
        @time begin
            for batch in 1:batches
                # TODO go back only once like in the notebook example:
                # https://gist.github.com/bchaber/48a309fbdad8753d2a60ce2f30da5e44

                x = train_x_batched[batch][1:196, :]
                result1 = forward(model, x)
                x = train_x_batched[batch][197:392, :]
                result2 = forward(model, x)
                x = train_x_batched[batch][393:588, :]
                result3 = forward(model, x)
                x = train_x_batched[batch][589:end, :]
                y = train_y_batched[batch]
                result4 = forward(model, x)

                loss, acc, C = AccuracyModule.loss_and_accuracy(result4, y)
                C = C ./ batches
                backward(model, C)
                push!(batch_acc, acc)
                push!(batch_loss, loss)
            end
        end
        for layer in model
            if isa(layer.layer, RecurrentNetworkModule.RNNCell)
                layer.layer.state = layer.layer.state0
            end
        end
        forward(model, train_x[1:196,:])
        forward(model, train_x[197:392,:])
        forward(model, train_x[393:588,:])
        result = forward(model, train_x[589:end,:])
        _, train_acc, _ = AccuracyModule.loss_and_accuracy(result, train_y)
        for layer in model
            if isa(layer.layer, RecurrentNetworkModule.RNNCell)
                layer.layer.state = layer.layer.state0
            end
        end
        forward(model, test_x[1:196,:])
        forward(model, test_x[197:392,:])
        forward(model, test_x[393:588,:])
        result = forward(model, test_x[589:end,:])
        _, test_acc, _ = AccuracyModule.loss_and_accuracy(result, test_y)
        @info epoch train_acc test_acc
        for layer in model
            if isa(layer.layer, RecurrentNetworkModule.RNNCell)
                layer.layer.state = layer.layer.state0
            end
        end
    end

#     plot(batch_loss, xlabel="Batch num", ylabel="loss", title="Loss over batches")
    plot(batch_acc, xlabel="Batch num", ylabel="acc", title="Accuracy over batches")
end
