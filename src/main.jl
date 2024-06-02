include("DataModule.jl")
include("RecurrentNetworkModule.jl")
include("DenseNetworkModule.jl")
include("AccuracyModule.jl")

using .DataModule, .RecurrentNetworkModule, .DenseNetworkModule, .AccuracyModule

function forward(model, input::Matrix{Float32})
    for layer in model
        input = layer(input)
    end
    return input
end

function train(model, data::Matrix{Float32}, data_y)
    result = forward(model, data)
    loss, acc, C = AccuracyModule.loss_and_accuracy(result, data_y)

    for layer in reverse(model)
        if isa(layer, DenseNetworkModule.Dense)
            C = DenseNetworkModule.back(layer, C)
        end
        if isa(layer, RecurrentNetworkModule.RNNCell)
            C = RecurrentNetworkModule.back(layer, C)
        end
    end
end

function main()
    # (in) => (out)
    rnn = RecurrentNetworkModule.RNN(784 => 64, tanh)
    rnn = RecurrentNetworkModule.RNN(784 => 10, tanh)
    dense1 = DenseNetworkModule.Dense(784 => 64, identity)
    dense2 = DenseNetworkModule.Dense(64 => 10, identity)
    dense = DenseNetworkModule.Dense(784 => 10, identity)
    model = [dense]
    batch_size = 100

    println("Loading training data...")
    train_x, train_y = DataModule.preprocess(:train; one_hot = true)

    train_x_batched = DataModule.batch(train_x, batch_size)
    train_y_batched = DataModule.batch(train_y, batch_size)

    println("Loading test data...")
    test_x, test_y = DataModule.preprocess(:test; one_hot = true)

    println("Training")
    batches = size(train_x_batched, 1)
    @time begin
        for batch in 1:batches
            train(model, train_x_batched[batch], train_y_batched[batch])
        end
    end
    result = forward(model, train_x)
    loss, acc, _ = AccuracyModule.loss_and_accuracy(result, train_y)
    println("Trained")
    @show loss
    @show acc

end
