include("AccuracyModule.jl")
using .AccuracyModule
using LinearAlgebra, StaticArrays
import Statistics: mean
# Types
abstract type GraphNode end
abstract type Operator <: GraphNode end

# Structs
struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable{O} <: GraphNode
    output :: O
    gradient :: Any
    name :: String
    Variable(output; name="") = new{typeof(output)}(output, nothing, name)
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    ScalarOperator(fun, inputs...; name="") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F, Inputs} <: Operator
    inputs :: Inputs
    output
    gradient
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun), typeof(inputs)}(inputs, nothing, nothing, name)
end

# Visitor
function visit(node::GraphNode, visited, order)
    if node ∉ visited
        push!(visited, node)
        push!(order, node)
    end
end

function visit(node::Operator, visited, order)
    if node ∉ visited
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
end

function topological_sort(head::GraphNode)
    visited = Set()
    order = Vector()
    visit(head, visited, order)
    return order
end

# Forward main
reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) =
    node.output = forward(node, [input.output for input in node.inputs]...)

function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end

# Backward main
update!(node::Constant, gradient) = nothing
update!(node::GraphNode, gradient) = if isnothing(node.gradient)
    node.gradient = gradient else node.gradient .+= gradient
end

function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    for node in reverse(order)
        backward!(node)
    end
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
    end
end

# Default useful operators
import Base: ^, *, +, -, /, sin, max, min, log, sum

+(x::GraphNode, y::GraphNode) = ScalarOperator(+, x, y)
forward(::ScalarOperator{typeof(+)}, x, y) = x + y
backward(::ScalarOperator{typeof(+)}, x, y, gradient) = (gradient, gradient)

-(x::GraphNode, y::GraphNode) = ScalarOperator(-, x, y)
forward(::ScalarOperator{typeof(-)}, x, y) = x - y
backward(::ScalarOperator{typeof(-)}, x, y, gradient) = (gradient, -gradient)

*(x::GraphNode, y::GraphNode) = ScalarOperator(*, x, y)
forward(::ScalarOperator{typeof(*)}, x, y) = x * y
backward(::ScalarOperator{typeof(*)}, x, y, gradient) = (y' * gradient, x' * gradient)

/(x::GraphNode, y::GraphNode) = ScalarOperator(/, x, y)
forward(::ScalarOperator{typeof(/)}, x, y) = x / y
backward(::ScalarOperator{typeof(/)}, x, y, gradient) = (gradient / y, gradient / y)

^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = x^n
backward(::ScalarOperator{typeof(^)}, x, n, gradient) = (gradient * n * x^(n - 1), gradient * log(abs(x)) * x^n)

sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = sin(x)
backward(::ScalarOperator{typeof(sin)}, x, gradient) = (gradient * cos(x))

log(x::GraphNode) = ScalarOperator(log, x)
forward(::ScalarOperator{typeof(log)}, x) = log(x)
backward(::ScalarOperator{typeof(log)}, x, gradient) = (gradient / x)

max(x::GraphNode, y::GraphNode) = ScalarOperator(max, x, y)
forward(::ScalarOperator{typeof(max)}, x, y) = max(x, y)
backward(::ScalarOperator{typeof(max)}, x, y, gradient) = (gradient * isless(y, x), gradient * isless(x, y))

min(x::GraphNode, y::GraphNode) = ScalarOperator(min, x, y)
forward(::ScalarOperator{typeof(min)}, x, y) = min(x, y)
backward(::ScalarOperator{typeof(min)}, x, y, gradient) = (gradient * isless(x, y), gradient * isless(y, x))

relu(x::GraphNode) = ScalarOperator(relu, x)
forward(::ScalarOperator{typeof(relu)}, x) = max(x, 0)
backward(::ScalarOperator{typeof(relu)}, x, gradient) = gradient * isless(0, x)

logistic(x::GraphNode) = ScalarOperator(logistic, x)
forward(::ScalarOperator{typeof(logistic)}, x) = 1 / (1 + exp(-x))
backward(::ScalarOperator{typeof(logistic)}, x, gradient) = gradient * exp(-x) / (1 + exp(-x))^2

# BROADCASTED

^(x::GraphNode, n::Number) = BroadcastedOperator(^, x, n)
forward(::BroadcastedOperator{typeof(^)}, x, n) = x .^ n
backward(::BroadcastedOperator{typeof(^)}, x, n, g) = tuple(g .* n .* x .^ (n - 1), nothing)

*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

relu(x::GraphNode) = BroadcastedOperator(relu, x)
forward(::BroadcastedOperator{typeof(relu)}, x) = return max.(x, zero(x))
backward(::BroadcastedOperator{typeof(relu)}, x, g) = return tuple(g .* (x .> 0))

log(x::GraphNode) = BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x) = 1 ./ (1 .+ exp.(-x))
backward(::BroadcastedOperator{typeof(log)}, x, g) = tuple(g .* exp.(x) ./ (1 .+ exp.(x)) .^ 2)

Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) =
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(vec(y .* 𝟏))
        Jy = diagm(vec(x .* 𝟏))
        tuple(Jx' * g, Jy' * g)
    end

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g, -g)

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)

sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) =
    let
        𝟏 =
        J = 𝟏'
        tuple(ones(length(x))'' * g)
    end

Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = x ./ y
function backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g)
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(𝟏 ./ y)
        Jy = (-x ./ y .^ 2)
        tuple(Jx' * g, Jy' * g)
    end
end

Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) =
    let
        Jx = diagm(isless.(y, x))
        Jy = diagm(isless.(x, y))
        tuple(Jx' * g, Jy' * g)
    end

dense_layer(x::GraphNode, w::GraphNode, b::GraphNode, f::Constant{T1}, df::Constant{T2}) where {T1 <: Function, T2 <: Function} = BroadcastedOperator(dense_layer, x, w, b, f, df)
forward(::BroadcastedOperator{typeof(dense_layer)}, x, w, b, f, df) = f.(w * x .+ b)
backward(::BroadcastedOperator{typeof(dense_layer)}, x, w, b, f, df, g) = let
    g = df.(w * x .+ b) .* g
    tuple(w' * g, g * x', sum(g, dims=2))
end

rnn_layer(U :: GraphNode, W :: GraphNode, h :: GraphNode, b :: GraphNode, x :: GraphNode, f::Constant{F1}, df::Constant{F2}) where {F1 <: Function, F2 <: Function} = BroadcastedOperator(rnn_layer, U, W, h, b, x, f, df)
forward(::BroadcastedOperator{typeof(rnn_layer)}, w, hw, state, b, x, f, df) = f.(w * x .+ hw * state .+ b)
backward(::BroadcastedOperator{typeof(rnn_layer)}, w, hw, state, b, x, f, df, g) = let
    dh = g .* df.(w * x .+ hw * state .+ b)
    tuple(dh * x', dh * state', hw' * dh, sum(dh, dims=2))
end