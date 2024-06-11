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
        if isa(node, Variable) && hasproperty(node, :gradient) && node.gradient != nothing
#             println(string("Update ", node.name, " ", sum(node.output), " ", sum(node.gradient)))
            if node.name == "state" || node.name == "x"
                node.output = node.gradient
                node.gradient = nothing
#                 println(string("UpdateSPECIAL ", node.name, " ", sum(node.output)))
            else
#                 println(string("Sizes: gradient: ", size(node.gradient), " output ", size(node.output)))
                node.gradient ./= batch_size
                node.output .-= lr * node.gradient
                node.gradient .= 0
            end
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
    state = Variable(zeros(Float32, size(x.output)), name = "state")

    r = rnn(x, wr, br, hwr, state)
    d = dense(r, wd, bd)
    e = cross_entropy_loss(d, y)
    graph = topological_sort(e)

    # RNN cell per 196 pixels and then sum the results?

    batch_loss = Float64[]
    println("Training")
    for epoch in 1:epochs
        @time for batch in batches
            reset_state!(graph)
            y.output = train_y_batched[batch]
            x.output = train_x_batched[batch][  1:196,:]
            state.output = zeros(Float32, 64, size(x.output, 2))
            forward!(graph)

            x.output = train_x_batched[batch][197:392,:]
            forward!(graph)

            x.output = train_x_batched[batch][393:588,:]
            forward!(graph)

            x.output = train_x_batched[batch][589:end,:]
            loss, acc = forward!(graph)

            push!(batch_loss, loss)
            backward!(graph)
            # Update gradientu raczej na samym końcu jak w Fluxie
            update_weights!(graph, 15e-3, batch_size)
        end
        reset_state!(graph)

        test_graph = topological_sort(d)

        y.output = test_y
        x.output = test_x[  1:196,:]
        state.output = zeros(Float32, 64, size(x.output, 2))
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
