include("AccuracyModule.jl")
using .AccuracyModule
using LinearAlgebra
using Flux
using NNlib
import Statistics: mean
# Types
abstract type GraphNode end
abstract type Operator <: GraphNode end

# Structs
struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
    name :: String
    Variable(output; name="?") = new(output, nothing, name)
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs
    output
    gradient
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
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

cross_entropy_loss(y_hat::GraphNode, y::GraphNode) = BroadcastedOperator(cross_entropy_loss, y_hat, y)
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y) = AccuracyModule.loss(y_hat, y)
backward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y, g) = let
    tuple(g .* AccuracyModule.softmax(y_hat) - y)
end

dense_layer(x::GraphNode, w::GraphNode, b::GraphNode) = BroadcastedOperator(dense_layer, x, w, b)
forward(::BroadcastedOperator{typeof(dense_layer)}, x, w, b) = w * x .+ b
backward(::BroadcastedOperator{typeof(dense_layer)}, x, w, b, g) = tuple(w' * g, g * x', sum(g, dims=2))

rnn_layer(x::GraphNode, w::GraphNode, b::GraphNode, hw::GraphNode, states::GraphNode) = BroadcastedOperator(rnn_layer, x, w, b, hw, states)
forward(o::BroadcastedOperator{typeof(rnn_layer)}, x, w, b, hw, states) = let
    if states == nothing
        state = zeros(Float32, size(w, 1), size(x, 2))
        o.inputs[5].output = Matrix{Float32}[]
    else
        state = last(states)
    end
    h = tanh.(w * x .+ hw * state .+ b)

    push!(o.inputs[5].output, reshape_cell_output(h, x))
    h
end
backward(::BroadcastedOperator{typeof(rnn_layer)}, x, w, b, hw, states, g) = let
    prev_state = zeros(Float32, size(states[1]))
    dw = zeros(Float32, size(w))
    dhw = zeros(Float32, size(hw))
    db = zeros(Float32, size(b))
    for state in reverse(states)
        zL = w * x .+ hw * state .+ b
        dp = state .+ hw * prev_state
        dtanh = (1 .- tanh.(zL).^2) .* dp .* g
        dw .+= dtanh * x'
        dhw .+= dtanh * state'
        db .+= mean(dtanh, dims=2)
        prev_state = state
    end

    tuple(w' * g, dw, db, dhw, nothing)
#     tuple(w' * g, g * x', sum(g, dims=2), g * state', nothing)
end

reshape_cell_output(h, x) = reshape(h, :, size(x)[2:end]...)

# backward(::BroadcastedOperator{typeof(rnn_layer)}, x, w, b, hw, state, g) = let
#     f = NNlib.fast_act(tanh_deriv, x)
#     xT = convert(Matrix{Float32}, x)
#     h = f(w * xT .+ hw * state .+ b) .* g
#     tuple(w' * g, h * x', sum(h, dims=2), h * state', nothing)
# end
