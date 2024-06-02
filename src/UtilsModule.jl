module UtilsModule
    using Random
    Random.seed!(123)
    nfan() = 1, 1 # fan_in, fan_out
    nfan(n) = 1, n # A vector is treated as a n×1 matrix
    nfan(n_out, n_in) = n_in, n_out # In case of Dense kernels: arranged as matrices
    nfan(dims::Tuple) = nfan(dims...)

    zeros32(size...) = zeros(Float32, size...)
    ones32(size...) = ones(Float32, size...)

    function glorot_uniform(dims::Integer...; gain::Real=1)
      scale = Float32(gain) * sqrt(24.0f0 / sum(nfan(dims...)))
      (rand(Float32, dims...) .- 0.5f0) .* scale
    end

    function _match_eltype(layer, ::Type{T}, x::AbstractArray{<:Union{AbstractFloat, Integer}}) where {T}
      convert(AbstractArray{T}, x)
    end

    function identity(x)
        return x
    end

    function identity_derivative(x)
        return 1
    end

    function tanh_derivative(x)
        mult = tanh(x) * tanh(x)
        return ones32(size(x)) - mult
    end


    function create_bias(weights::AbstractArray, bias::Bool, dims::Integer...)
      bias ? fill!(similar(weights, dims...), 0) : false
    end

    function create_bias(weights::AbstractArray, bias::AbstractArray, dims::Integer...)
      size(bias) == dims || throw(DimensionMismatch("expected bias of size $(dims), got size $(size(bias))"))
      convert(AbstractArray{_eltype(weights)}, bias)
    end

    fast_act(f::F, ::AbstractArray = 1:0) where {F<:Function} = f
end