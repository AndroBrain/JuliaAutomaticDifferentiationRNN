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
      println(size(result))
      return result
    end

    function (a::Dense)(loss, x::AbstractVecOrMat)
      derivative_activation = a.σ_d(x)
      a.weight = a.weight * derivative_activation
      return a.weight * derivative_activation
    end

    function back(a::Dense, C::Matrix{Float32})
        prev = a.prev_input
#         prev = UtilsModule.ones32(size(a.weight)) * a.prev_input
        println(string("Prev_input: ", size(prev))) # 64, 60000
        deriv_a = 1 # identity_derivative is always 1
        println(string("deriv_a: ", size(deriv_a))) # 1
        println(string("C: ", size(C))) # 10, 60000
        weight_gradient = prev * deriv_a * transpose(C)
        println(string("Weight gradient: ", size(weight_gradient)))
        println(string("a weight: ", size(a.weight)))
        a.weight .-= transpose(weight_gradient)

#         bias_gradient = deriv_a * mean(C)
#         println(string("Bias gradient", size(bias_gradient)))
#         println(string("b weight: ", size(a.bias)))
#         a.bias .-= bias_gradient
    end

    function show(l::Dense)
      print("Dense(", size(l.weight, 2), " => ", size(l.weight, 1))
      l.σ == identity || print(", ", l.σ)
      l.bias == false && print("; bias=false")
      print(")\n")
    end
end
