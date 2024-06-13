include("Graph.jl")
include("DataModule.jl")
include("UtilsModule.jl")
include("AccuracyModule.jl")
include("GradientOptimizersModule.jl")

using .DataModule, .UtilsModule, .AccuracyModule, .GradientOptimizersModule
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

function update_weights!(graph::Vector, optimizer::GradientOptimizersModule.GradientOptimizer)
    for node in graph
        if isa(node, Variable) && (node.name == "states" || node.name == "x")
            node.output = nothing
            node.gradient = nothing
        elseif isa(node, Variable) && hasproperty(node, :gradient) && node.gradient != nothing
            node.output .-= optimizer(node.gradient)
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

    epochs = 5

    x = Variable([0.], name="x")
    y = Variable([0.], name="y")

    wd = Variable(UtilsModule.glorot_uniform(10, 64))
    bd = Variable(UtilsModule.glorot_uniform(10, ))

    wr = Variable(UtilsModule.glorot_uniform(64, 196))
    br = Variable(UtilsModule.glorot_uniform(64, ))
    hwr = Variable(UtilsModule.glorot_uniform(64, 64))
    states = Variable(nothing, name = "states")

    optimizer = GradientOptimizersModule.Descent(15e-3)

    r = rnn_layer(x, wr, br, hwr, states)
    d = dense_layer(r, wd, bd)
    graph = topological_sort(d)

    batch_loss = Float64[]
    println("Training")
    for epoch in 1:epochs
        batches = randperm(size(train_x_batched, 1))
        @time for batch in batches
            reset_state!(graph)
            states.output = nothing
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
            gradient = AccuracyModule.get_gradient(result, y.output) ./ batch_size
            backward!(graph, seed=gradient)
            update_weights!(graph, optimizer)
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

        states.output = zeros(Float32, size(x.output))
        @show epoch loss acc
    end
    plot(batch_loss, xlabel="Batch num", ylabel="loss", title="Loss over batches")
end
