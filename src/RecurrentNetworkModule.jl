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

    RNNCell((in, out)::Pair, activation = tanh; init = UtilsModule.glorot_uniform, initb = UtilsModule.zeros32, init_state = UtilsModule.zeros32) =
      RNNCell(activation, init(out, in), init(out, out), initb(out), init_state(out,1), init_state(out, 1), UtilsModule.zeros32(in, in))

    RNN(a...; ka...) = RNNCell(a...; ka...)

    # Forward propagation
    function (m::RNNCell)(x::Matrix{Float32})
      m.prev_input = x
#       m.state = m.activation.(m.input_weights*x .+ m.hidden_weights * m.state .+ b)
      m.state = m.activation.(m.input_weights*x .+ m.bias)
      return m.state
    end

    function back(a::RNNCell, C::Matrix{Float32})
        z_l = a.input_weights * a.prev_input
        der_z = 1 .- tanh.(z_l).^2
#         @info "data: " size(der_z) size(a.state) size(a.prev_input) size(C) size(a.Wi)
        # TODO optimize by calculating der_z .*
        fast_calc = der_z .* C
        gradient_weights = fast_calc * a.prev_input'
        gradient_bias = sum(fast_calc, dims=2)
#         println(string("gradient_weights: ", size(gradient_weights)))
#         @show sum(gradient_weights)

        # TODO update hidden state weights
        a.input_weights .-= gradient_weights
        a.bias .-= gradient_bias
    end
end
