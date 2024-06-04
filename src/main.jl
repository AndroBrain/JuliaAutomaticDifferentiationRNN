include("DataModule.jl")
# include("RecurrentNetworkModule.jl")
# include("DenseNetworkModule.jl")
include("AccuracyModule.jl")
include("GradientOptimizersModule.jl")
include("ModelModule.jl")

using .DataModule, .AccuracyModule, .GradientOptimizersModule, .ModelModule
using Plots, Random, Statistics

struct LayerWrapper
    layer
    optimizer::GradientOptimizer
end

function load_data(batch_size)
    println("Loading train data...")
    train_x, train_y = DataModule.preprocess(:train; one_hot = true)

    train_x_batched = DataModule.batch(train_x, batch_size)
    train_y_batched = DataModule.batch(train_y, batch_size)

    println("Loading test data...")
    test_x, test_y = DataModule.preprocess(:test; one_hot = true)
    return train_x, train_y, train_x_batched, train_y_batched, test_x, test_y
end

function train(model, epochs, mini_batches, train_x_batched, train_y_batched, train_x, train_y, test_x, test_y)
    batch_acc = Float64[]
    batch_loss = Float64[]
    batches = randperm(size(train_x_batched, 1))
    input_size = size(train_x_batched[1], 1)
    mini_batch_size = convert(Int64, input_size / mini_batches)
    for epoch in 1:epochs
        @time begin
            for batch in batches
                zero_state(model)

                result = mini_batch_forward(model, train_x_batched[batch], mini_batches, mini_batch_size)

                y = train_y_batched[batch]
                loss, acc, C = AccuracyModule.loss_and_accuracy(result, y)
                C = C ./ mini_batches
                backward(model, C)
                push!(batch_acc, acc)
                push!(batch_loss, loss)
            end
        end
        zero_state(model)
        result = mini_batch_forward(model, train_x, mini_batches, mini_batch_size)
        train_loss, train_acc, _ = AccuracyModule.loss_and_accuracy(result, train_y)

        zero_state(model)
        result = mini_batch_forward(model, test_x, mini_batches, mini_batch_size)
        test_loss, test_acc, _ = AccuracyModule.loss_and_accuracy(result, test_y)

        @info epoch train_acc train_loss test_acc test_loss
    end
    plot(batch_loss, xlabel="Batch num", ylabel="loss", title="Loss over batches")
#     plot(batch_acc, xlabel="Batch num", ylabel="acc", title="Accuracy over batches")
end

function main()
    batch_size = 100
    mini_batches = 4
    train_x, train_y, train_x_batched, train_y_batched, test_x, test_y = load_data(batch_size)

    println("Training...")
    # (in) => (out)
    descent = Descent(15e-3)
    rnn = LayerWrapper(RecurrentNetworkModule.RNN(196 => 64, tanh), descent)
    dense = LayerWrapper(DenseNetworkModule.Dense(64 => 10, identity), descent)

    model = [rnn, dense]
    epochs = 5
    train(model, epochs, mini_batches, train_x_batched, train_y_batched, train_x, train_y, test_x, test_y)
end
