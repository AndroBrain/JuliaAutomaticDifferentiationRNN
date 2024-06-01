include("DataModule.jl")
include("RecurrentNetworkModule.jl")
include("DenseNetworkModule.jl")
include("AccuracyModule.jl")

using .DataModule, .RecurrentNetworkModule, .DenseNetworkModule, .AccuracyModule

function forward(model::Tuple, input::Matrix{Float32})
    for layer in model
        input = layer(input)
    end
    return input
end

function train(model::Tuple, data::Matrix{Float32}, data_y)
#     data1 = data[1:196, :]
#     data2 = data[197:392, :]
#     data3 = data[393:588, :]
#     data4 = data[589:end, :]
#     a1 = forward(model, data1)
#     a2 = forward(model, data2)
#     a3 = forward(model, data3)
#     a4 = forward(model, data4)
    result = forward(model, data)
    C = (2 * (result - data_y) / size(data_y)[2])

    loss, acc, gradient_loss = AccuracyModule.loss_and_accuracy(result, data_y)
    @show loss
    @show acc

    for layer in model
        if isa(layer, DenseNetworkModule.Dense)
            println("Backprop")
            DenseNetworkModule.back(layer, C)
        end
    end

    result = forward(model, data)

    loss, acc, gradient_loss = AccuracyModule.loss_and_accuracy(result, data_y)
    @show loss
    @show acc
#
#     a1 = forward(model, data1)
#     a2 = forward(model, data2)
#     a3 = forward(model, data3)
#     a4 = forward(model, data4)
end

function main()
    # (in) => (out)
    rnn = RecurrentNetworkModule.RNN((784) => 64, tanh)
    dense = DenseNetworkModule.Dense(64 => 10, identity)
    model = (rnn, dense)

    println("Loading training data...")
    train_x, train_y = DataModule.preprocess(:train; one_hot = true)

    println("Loading test data...")
    test_x, test_y = DataModule.preprocess(:test; one_hot = true)

    println("Training")
    @time begin
        train(model, train_x, train_y)
    end

#     println("Testing")
#     rnn.state = rnn.state0
#     test_result = train(model, test_x)
#     loss, acc = AccuracyModule.loss_and_accuracy(test_result, test_y)
#     @show loss
#     @show acc

end
