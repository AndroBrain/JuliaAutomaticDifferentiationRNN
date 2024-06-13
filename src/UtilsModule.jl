module UtilsModule
    using Random
    Random.seed!(123)

    nfan() = 1, 1
    nfan(n) = 1, n
    nfan(n_out, n_in) = n_in, n_out
    nfan(dims::Tuple) = nfan(dims...)

    zeros32(size...) = zeros(Float32, size...)
    ones32(size...) = ones(Float32, size...)

    function glorot_uniform(dims::Integer...; gain::Real=1)
      scale = Float32(gain) * sqrt(24.0f0 / sum(nfan(dims...)))
      (rand(Float32, dims...) .- 0.5f0) .* scale
    end

    function identity(x)
        return x
    end

    function identity_derivative(x)
        return 1
    end

    function tanh_derivative(x)
        return ones(Float32, size(x)) - tanh.(x).^2
    end
end