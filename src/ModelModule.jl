module ModelModule
    include("RecurrentNetworkModule.jl")
    include("DenseNetworkModule.jl")
    using .RecurrentNetworkModule, .DenseNetworkModule
    export zero_state, backward, forward, mini_batch_forward, DenseNetworkModule, RecurrentNetworkModule

    function zero_state(model)
        for layer in model
            if isa(layer.layer, RecurrentNetworkModule.RNNCell)
                layer.layer.state = layer.layer.state0
            end
        end
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

    function mini_batch_forward(model, x, mini_batches, mini_batch_size)
        result = nothing
        for mini_batch in 1:mini_batches
            result = forward(model, x[(mini_batch - 1) * mini_batch_size + 1:mini_batch * mini_batch_size, :])
        end
        return result
    end
end
