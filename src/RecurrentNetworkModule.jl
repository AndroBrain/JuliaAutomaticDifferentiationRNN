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
#       m.state = σ.(Wi*x .+ Wh*m.state .+ b)
      m.state = σ.(Wi*x .+ b)
      return m.state
    end

    function back(a::RNNCell, C::Matrix{Float32})
        z_l = a.Wi * a.prev_input
        der_z = 1 .- tanh.(z_l).^2
#         @info "data: " size(der_z) size(a.state) size(a.prev_input) size(C) size(a.Wi)
        gradient_weights = der_z .* C * a.prev_input'
#         println(string("gradient_weights: ", size(gradient_weights)))
#         @show sum(gradient_weights)

        a.Wi .-= gradient_weights
    end

    function show(l::RNNCell)
      print("RNNCell(", size(l.Wi, 2), " => ", size(l.Wi, 1))
      l.σ == identity || print(", ", l.σ)
      print(")\n")
    end
end
