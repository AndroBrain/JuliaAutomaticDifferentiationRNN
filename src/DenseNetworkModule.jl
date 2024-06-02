# Utils
module DenseNetworkModule
    include("UtilsModule.jl")
    using .UtilsModule
    using Statistics: mean

    mutable struct Dense{F, FA, M<:AbstractMatrix, B, P}
      weight::M
      bias::B
      σ::F
      σ_d::FA
      prev_input::P
      function Dense(W::M, bias = true, σ::F = identity,σ_d::FA = identity_derivative, prev_input::P = 1) where {M<:AbstractMatrix, F, FA, P}
        b = UtilsModule.create_bias(W, bias, size(W,1))
        new{F, FA,M,typeof(b), P}(W, b, σ, σ_d, prev_input)
      end
    end

    function Dense((in, out)::Pair{<:Integer, <:Integer}, σ = UtilsModule.identity;
      σ_d = UtilsModule.identity_derivative, init = UtilsModule.glorot_uniform, bias = true)
      Dense(init(out, in), bias, σ, σ_d, UtilsModule.zeros32(in, in))
    end

    function (a::Dense)(x::AbstractVecOrMat)
      a.prev_input = x
      σ = UtilsModule.fast_act(a.σ, x)
    #   xT = _match_eltype(a, x)  # TODO try to make it work or remove
      result = σ.(a.weight * x .+ a.bias)
      return result
    end

    function back(a::Dense, C::Matrix{Float32})
        prev_weight = a.weight

        gradient_weights = C * a.prev_input'
        gradient_biases = sum(C, dims=2)
        a.weight .-= gradient_weights
        a.bias .-= gradient_biases
        return prev_weight' * C
    end

    function show(l::Dense)
      print("Dense(", size(l.weight, 2), " => ", size(l.weight, 1))
      l.σ == identity || print(", ", l.σ)
      l.bias == false && print("; bias=false")
      print(")\n")
    end
end
