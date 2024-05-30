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

function train(model::Tuple, data::Matrix{Float32})
    forward(model, data[1:196,:])
    forward(model, data[197:392,:])
    forward(model, data[393:588,:])
    return forward(model, data[589:end,:])
end

function main()
    # (in) => (out)
    dense = DenseNetworkModule.Dense(64 => 10, identity)
    rnn = RecurrentNetworkModule.RNN((196) => 64, tanh)
    model = (rnn, dense)

    println("Loading training data...")
    train_x, train_y = DataModule.preprocess(:train; one_hot = true)

    println("Loading test data...")
    test_x, test_y = DataModule.preprocess(:test; one_hot = true)

    println("Training")
    @time begin
        train_result = train(model, train_x)
    end
    loss, acc = AccuracyModule.loss_and_accuracy(train_result, train_y)
    @show loss
    @show acc

    println("Testing")
    rnn.state = rnn.state0
    test_result = train(model, test_x)
    loss, acc = AccuracyModule.loss_and_accuracy(test_result, test_y)
    @show loss
    @show acc

end
