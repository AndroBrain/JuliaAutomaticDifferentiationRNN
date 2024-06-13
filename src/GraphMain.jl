include("Graph.jl")
include("DataModule.jl")
include("UtilsModule.jl")
include("AccuracyModule.jl")

using .DataModule, .UtilsModule, .AccuracyModule
using Random, Plots

function load_data(batch_size)
    println("Loading train data...")
    train_x, train_y = DataModule.preprocess(:train; one_hot = true)

    train_x_batched = DataModule.batch(train_x, batch_size)
    train_y_batched = DataModule.batch(train_y, batch_size)

    println("Loading test data...")
    test_x, test_y = DataModule.preprocess(:test; one_hot = true)
    return train_x, train_y, train_x_batched, train_y_batched, test_x, test_y
end

function update_weights!(graph::Vector, lr::Float64, batch_size::Int64)
    for node in graph
        if isa(node, Variable) && (node.name == "state" || node.name == "x")
                node.output = nothing
                node.gradient = nothing
        elseif isa(node, Variable) && hasproperty(node, :gradient) && node.gradient != nothing
#             println(string("Update ", node.name, " ", sum(node.output), " ", sum(node.gradient)))
#                 println(string("Sum1: ", node.name, " gradient: ", sum(node.gradient), " output ", sum(node.output)))
            node.gradient ./= batch_size
            node.output .-= lr * node.gradient
#                 println(string("Sum2: ", node.name, " gradient: ", sum(node.gradient), " output ", sum(node.output)))
            node.gradient .= 0
        end
    end
end

function reset_state!(graph::Vector)
    for node in graph
        if isa(node, Variable) && hasproperty(node, :gradient) && node.gradient != nothing
            node.gradient .= 0
        end
    end
end

function main()
    batch_size = 100
    train_x, train_y, train_x_batched, train_y_batched, test_x, test_y = load_data(batch_size)
    batches = randperm(size(train_x_batched, 1))

    epochs = 5

    x = Variable([0.], name="x")
    y = Variable([0.], name="y")

    wd = Variable(UtilsModule.glorot_uniform(10, 64), name="wd")
    bd = Variable(UtilsModule.glorot_uniform(10, ), name="bd")

    wr = Variable(UtilsModule.glorot_uniform(64, 196), name = "wr")
    br = Variable(UtilsModule.glorot_uniform(64, ), name = "br")
    hwr = Variable(UtilsModule.glorot_uniform(64, 64), name = "hwr")
    state = Variable(nothing, name = "state")

    r = rnn_layer(x, wr, br, hwr, state)
    d = dense_layer(r, wd, bd)
    graph = topological_sort(d)

    # RNN cell per 196 pixels and then sum the results?

    batch_loss = Float64[]
    println("Training")
    for epoch in 1:epochs
        @time for batch in batches
            reset_state!(graph)
            state.output = nothing
            y.output = train_y_batched[batch]
            x.output = train_x_batched[batch][  1:196,:]
            forward!(graph)

            x.output = train_x_batched[batch][197:392,:]
            forward!(graph)

            x.output = train_x_batched[batch][393:588,:]
            forward!(graph)

            x.output = train_x_batched[batch][589:end,:]
            result = forward!(graph)

            loss, acc, _ = AccuracyModule.loss_and_accuracy(result, train_y_batched[batch])
            push!(batch_loss, loss)
            gradient = AccuracyModule.get_gradient(result, y.output)
            backward!(graph, seed=gradient)
            # Update gradientu raczej na samym końcu jak w Fluxie
            update_weights!(graph, 15e-3, batch_size)
        end
        test_graph = topological_sort(d)

        y.output = test_y
        x.output = test_x[  1:196,:]
        reset_state!(test_graph)
        forward!(test_graph)

        x.output = test_x[197:392,:]
        forward!(test_graph)

        x.output = test_x[393:588,:]
        forward!(test_graph)

        x.output = test_x[589:end,:]
        result = forward!(test_graph)

        loss, acc, _ = AccuracyModule.loss_and_accuracy(result, test_y)

        state.output = zeros(Float32, size(x.output))
        @show epoch loss acc
    end
    plot(batch_loss, xlabel="Batch num", ylabel="loss", title="Loss over batches")
end
