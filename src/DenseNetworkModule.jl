# Utils
module DenseNetworkModule
    include("UtilsModule.jl")
    using .UtilsModule

    struct Dense{F, M<:AbstractMatrix, B}
      weight::M
      bias::B
      σ::F
      function Dense(W::M, bias = true, σ::F = identity) where {M<:AbstractMatrix, F}
        b = UtilsModule.create_bias(W, bias, size(W,1))
        new{F,M,typeof(b)}(W, b, σ)
      end
    end

    function Dense((in, out)::Pair{<:Integer, <:Integer}, σ = UtilsModule.identity;
                   init = UtilsModule.glorot_uniform, bias = true)
      Dense(init(out, in), bias, σ)
    end

    function (a::Dense)(x::AbstractVecOrMat)
      σ = UtilsModule.fast_act(a.σ, x)
    #   xT = _match_eltype(a, x)  # TODO try to make it work or remove
      return σ.(a.weight * x .+ a.bias)
    end

    function show(l::Dense)
      print("Dense(", size(l.weight, 2), " => ", size(l.weight, 1))
      l.σ == identity || print(", ", l.σ)
      l.bias == false && print("; bias=false")
      print(")\n")
    end
end

# ************************************************************************************************************************************************

# include("DataLoader.jl")
# include("RecurrentNetworkModule.jl")
#
# using .DataLoader, .RecurrentNetworkModule
#
# function main()
#     # (int) => (out)
#     dense = Dense(64 => 10, identity)
#     show(dense)
#     rnn = RecurrentNetworkModule.RNN((196) => 64, tanh)
#     RecurrentNetworkModule.show(rnn.cell)
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
#     dense(rnn.state)
#     println("Calculating #392")
#     rnn.state = rnn.cell(rnn.state, train_x[197:392,:])
#     dense(rnn.state)
#     println("Calculating #588")
#     rnn.state = rnn.cell(rnn.state, train_x[393:588,:])
#     dense(rnn.state)
#     println("Calculating #end")
#     rnn.state = rnn.cell(rnn.state, train_x[589:end,:])
#     dense(rnn.state)
# end
