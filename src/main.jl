include("DataModule.jl")
include("RecurrentNetworkModule.jl")
include("DenseNetworkModule.jl")
include("AccuracyModule.jl")

using .DataModule, .RecurrentNetworkModule, .DenseNetworkModule, .AccuracyModule

function main()
    # (in) => (out)
    dense = DenseNetworkModule.Dense(64 => 10, identity)
    rnn = RecurrentNetworkModule.RNN((196) => 64, tanh)

    println("Loading training data...")
    train_x, train_y = DataModule.preprocess(:train; one_hot = true)

    println("Loading test data...")
    test_x, test_y = DataModule.preprocess(:test; one_hot = true)

    println("Training")
#     println("Calculating #196")
    rnn.state = rnn.cell(rnn.state, train_x[1:196,:])
    dense(rnn.state)
#     println("Calculating #392")
    rnn.state = rnn.cell(rnn.state, train_x[197:392,:])
    dense(rnn.state)
#     println("Calculating #588")
    rnn.state = rnn.cell(rnn.state, train_x[393:588,:])
    dense(rnn.state)
#     println("Calculating #end")
    rnn.state = rnn.cell(rnn.state, train_x[589:end,:])
    train_result = dense(rnn.state)

    loss, acc = AccuracyModule.loss_and_accuracy(train_result, train_y)

    println("Train results")

    @show loss
    @show acc

    rnn.state = RecurrentNetworkModule.RNN((196) => 64, tanh).state

#     println("Calculating #196")
    rnn.state = rnn.cell(rnn.state, test_x[1:196,:])
    dense(rnn.state)
#     println("Calculating #392")
    rnn.state = rnn.cell(rnn.state, test_x[197:392,:])
    dense(rnn.state)
#     println("Calculating #588")
    rnn.state = rnn.cell(rnn.state, test_x[393:588,:])
    dense(rnn.state)
#     println("Calculating #end")
    rnn.state = rnn.cell(rnn.state, test_x[589:end,:])
    test_result = dense(rnn.state)

    loss, acc = AccuracyModule.loss_and_accuracy(test_result, test_y)

    println("Test results")

    @show loss
    @show acc

end
