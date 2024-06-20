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
        if isa(node, Variable)
            if node.gradient != nothing
                node.output .-= optimizer(node.gradient)
                node.gradient .= 0
            elseif node.gradient == nothing
                node.output = nothing
            end
        end
    end
end

function main()
    batch_size = 100
    train_x, train_y, train_x_batched, train_y_batched, test_x, test_y = load_data(batch_size)

    epochs = 5

    x1 = Variable([0.], name = "x")
    x2 = Variable([0.], name = "x")
    x3 = Variable([0.], name = "x")
    x4 = Variable([0.], name = "x")

    wd = Variable(UtilsModule.glorot_uniform(10, 64))
    bd = Variable(UtilsModule.glorot_uniform(10, ))
    fd = Constant(UtilsModule.identity)
    dfd = Constant(UtilsModule.identity_deriv)

    wr = Variable(UtilsModule.glorot_uniform(64, 196))
    hwr = Variable(UtilsModule.glorot_uniform(64, 64))
    br = Variable(UtilsModule.glorot_uniform(64, ))
    fr = Constant(tanh)
    dfr = Constant(UtilsModule.tanh_deriv)

    state0 = Variable(zeros(64))

    optimizer = GradientOptimizersModule.Descent(15e-3)

    r1 = rnn_layer(wr, hwr, state0, br, x1)
    r2 = rnn_layer(wr, hwr, r1, br, x2)
    r3 = rnn_layer(wr, hwr, r2, br, x3)
    r4 = rnn_layer(wr, hwr, r3, br, x4)
    d = dense_layer(r4, wd, bd, fd, dfd)
    graph = topological_sort(d)

    batch_loss = Float64[]
    println("Training")
    for epoch in 1:epochs
        batches = randperm(size(train_x_batched, 1))
        @time for batch in batches
            state0.output = zeros(64, 100)
            x1.output = train_x_batched[batch][  1:196,:]
            x2.output = train_x_batched[batch][197:392,:]
            x3.output = train_x_batched[batch][393:588,:]
            x4.output = train_x_batched[batch][589:end,:]
            result = forward!(graph)

            loss = AccuracyModule.loss(result, train_y_batched[batch])
            push!(batch_loss, loss)
            gradient = AccuracyModule.gradient(result, train_y_batched[batch]) ./ batch_size
            backward!(graph, seed=gradient)
            update_weights!(graph, optimizer)
        end
        state0.output = zeros(64, 10000)
        test_graph = topological_sort(d)

        x1.output = test_x[  1:196,:]
        x2.output = test_x[197:392,:]
        x3.output = test_x[393:588,:]
        x4.output = test_x[589:end,:]
        result = forward!(test_graph)

        loss = AccuracyModule.loss(result, test_y)
        acc = AccuracyModule.accuracy(result, test_y)

        @show epoch loss acc
    end
    plot(batch_loss, xlabel="Batch num", ylabel="loss", title="Loss over batches")
end
