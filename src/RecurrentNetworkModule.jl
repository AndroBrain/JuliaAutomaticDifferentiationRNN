module RecurrentNetworkModule
    include("UtilsModule.jl")
    using .UtilsModule

    mutable struct Recur{T,S}
      cell::T
      state::S
    end

    function (m::Recur)(x)
      m.state, y = m.cell(m.state, x)
      return y
    end

    struct RNNCell{F,I,H,V,S}
      σ::F
      Wi::I
      Wh::H
      b::V
      state0::S
    end

    function (m::RNNCell{F,I,H,V,<:AbstractMatrix{T}})(h, x::Matrix{Float32}) where {F,I,H,V,T}
      Wi, Wh, b = m.Wi, m.Wh, m.b
      σ = UtilsModule.fast_act(m.σ, x)
      xT = UtilsModule._match_eltype(m, T, x)
      h = σ.(Wi*xT .+ Wh*h .+ b)
      return h
    end

    RNNCell((in, out)::Pair, σ = tanh; init = UtilsModule.glorot_uniform, initb = UtilsModule.zeros32, init_state = UtilsModule.zeros32) =
      RNNCell(σ, init(out, in), init(out, out), initb(out), init_state(out,1))

    RNN(a...; ka...) = Recur(RNNCell(a...; ka...))
    Recur(m::RNNCell) = Recur(m, m.state0)

    function show(l::RNNCell)
      print("RNNCell(", size(l.Wi, 2), " => ", size(l.Wi, 1))
      l.σ == identity || print(", ", l.σ)
      print(")\n")
    end
end

# ************************************************************************************************************************************************

# include("DataLoader.jl")
#
# using .DataLoader
#
# function main()
#     # (int) => (out)
#     rnn = RNN((196) => 64, tanh)
#     show(rnn.cell)
#
#     println("Loading data...")
#     # Load and preprocess train data
#     train_features, train_labels = DataLoader.load(:train)
#     train_x, train_y = DataLoader.preprocess(train_features, train_labels; one_hot = true)
#
#     # Load and preprocess test data
#     test_features, test_labels = DataLoader.load(:test)
#     test_x, test_y = DataLoader.preprocess(test_features, test_labels; one_hot = true)
#
#     println("Calculating #196")
#     rnn.state = rnn.cell(rnn.state, train_x[1:196,:])
#     println("Calculating #392")
#     rnn.state = rnn.cell(rnn.state, train_x[197:392,:])
#     println("Calculating #588")
#     rnn.state = rnn.cell(rnn.state, train_x[393:588,:])
#     println("Calculating #end")
#     rnn.state = rnn.cell(rnn.state, train_x[589:end,:])
#
# end
