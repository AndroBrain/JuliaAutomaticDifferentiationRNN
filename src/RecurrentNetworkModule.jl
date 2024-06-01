module RecurrentNetworkModule
    include("UtilsModule.jl")
    using .UtilsModule

    mutable struct RNNCell{F,I,H,V,S,S0,P}
      σ::F
      Wi::I
      Wh::H
      b::V
      state::S
      state0::S0
      prev_input::P
    end

    RNNCell((in, out)::Pair, σ = tanh; init = UtilsModule.glorot_uniform, initb = UtilsModule.zeros32, init_state = UtilsModule.zeros32) =
      RNNCell(σ, init(out, in), init(out, out), initb(out), init_state(out,1), init_state(out, 1), UtilsModule.zeros32(in, in))

    RNN(a...; ka...) = RNNCell(a...; ka...)

    # Forward propagation
    function (m::RNNCell)(x::Matrix{Float32})
      m.prev_input = x
      Wi, Wh, b = m.Wi, m.Wh, m.b
      σ = UtilsModule.fast_act(m.σ, x)
      m.state = σ.(Wi*x .+ Wh*m.state .+ b)
      return m.state
    end

    function back(m::RNNCell, C::Matrix{Float32})
        prev = UtilsModule.ones32(size(m.Wi)) * m.prev_input
        println(string("Prev_input: ", size(prev))) # 784, 60000
        x = m.Wi * m.prev_input
        deriv_a = 1 .- tanh.(x).^2
        println(string("deriv_a: ", size(deriv_a))) # 64, 60000
        println(string("C: ", size(C))) # 10, 60000
        println(string("mult1: ", size(prev.*deriv_a)))
        weight_gradient = prev .* deriv_a .* C
        println("Weight gradient: " + size(weight_gradient))
    end

    function show(l::RNNCell)
      print("RNNCell(", size(l.Wi, 2), " => ", size(l.Wi, 1))
      l.σ == identity || print(", ", l.σ)
      print(")\n")
    end
end
