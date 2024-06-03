# Utils
module DenseNetworkModule
    include("UtilsModule.jl")
    using .UtilsModule
    using Statistics: mean

    mutable struct Dense{M<:AbstractMatrix, B, P}
      weight::M
      bias::B
      activation::Function
      prev_input::P
      function Dense(W::M, bias = true, activation::F = identity, prev_input::P = 1) where {M<:AbstractMatrix, F, P}
        b = UtilsModule.create_bias(W, bias, size(W,1))
        new{M,typeof(b), P}(W, b, activation, prev_input)
      end
    end

    function Dense((in, out)::Pair{<:Integer, <:Integer}, activation = UtilsModule.identity;
      init = UtilsModule.glorot_uniform, bias = true)
      Dense(init(out, in), bias, activation, UtilsModule.zeros32(in, in))
    end

    function (a::Dense)(x::AbstractVecOrMat)
      a.prev_input = x
    #   xT = _match_eltype(a, x)  # TODO try to make it work or remove
      result = a.activation.(a.weight * x .+ a.bias)
      return result
    end

    function back(a::Dense, C::AbstractVecOrMat)
        # TODO add activation derivative function
        prev_weight = a.weight
        gradient_weights = C * a.prev_input'
        gradient_biases = sum(C, dims=2)
        a.weight .-= gradient_weights
        a.bias .-= gradient_biases
        return prev_weight' * C
    end
end
