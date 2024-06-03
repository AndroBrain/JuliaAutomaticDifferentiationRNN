module RecurrentNetworkModule
    include("UtilsModule.jl")
    using .UtilsModule

    mutable struct RNNCell{I,H,V,S,S0,P}
      activation::Function
      input_weights::I
      hidden_weights::H
      bias::V
      state::S
      state0::S0
      prev_input::P
    end

    function RNNCell((in, out)::Pair, activation = tanh; init = UtilsModule.glorot_uniform, init_state = UtilsModule.zeros32)
      input_weights = init(out, in)
      bias = UtilsModule.create_bias(input_weights, true, size(input_weights, 1))
      RNNCell(activation, input_weights, init(out, out), bias, init_state(out,1), init_state(out, 1), UtilsModule.zeros32(in, in))
    end

    RNN(a...; ka...) = RNNCell(a...; ka...)

    # Forward propagation
    function (m::RNNCell)(x::AbstractVecOrMat)
      m.prev_input = x
      m.state = m.activation.(m.input_weights * x .+ m.hidden_weights * m.state .+ m.bias)
      return m.state
    end

    function back(a::RNNCell, C::AbstractVecOrMat)
        z_l = a.input_weights * a.prev_input
        der_z = 1 .- tanh.(z_l).^2
        fast_calc = der_z .* C
        gradient_weights = fast_calc * a.prev_input'
        gradient_hidden_weights = fast_calc * a.state'
        gradient_bias = sum(fast_calc, dims=2)

        a.input_weights .-= gradient_weights
        a.hidden_weights .-= gradient_hidden_weights
        a.bias .-= gradient_bias
    end
end
