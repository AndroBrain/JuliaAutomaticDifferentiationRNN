include("AccuracyModule.jl")
using .AccuracyModule
using LinearAlgebra
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
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y) = AccuracyModule.loss_acc(y_hat, y)
backward(node::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y, g) = tuple(AccuracyModule.softmax(y_hat) - y)

dense(x::GraphNode, w::GraphNode, b::GraphNode) = BroadcastedOperator(dense, x, w, b)
forward(::BroadcastedOperator{typeof(dense)}, x, w, b) = w * x .+ b
backward(::BroadcastedOperator{typeof(dense)}, x, w, b, g) = tuple(w' * g, g * x', sum(g, dims=2))

rnn(x::GraphNode, w::GraphNode, b::GraphNode, hw::GraphNode, state::GraphNode) = BroadcastedOperator(rnn, x, w, b, hw, state)
forward(o::BroadcastedOperator{typeof(rnn)}, x, w, b, hw, state) = let
    o.inputs[5].output = o.inputs[5].output .+ tanh.(w * x .+ hw * state)
#     o.inputs[5].output = tanh.(w * x .+ hw * state)
    # TODO consider trying to make an array of hidden functions, even consider array of weights
    tanh.(w * x .+ hw * state .+ b)
end
backward(::BroadcastedOperator{typeof(rnn)}, x, w, b, hw, state, g) = let
    g = (1 .- tanh.(w * x).^2) .* g
#     println(string("Backward ", sum(w' * g), " ", sum(g * x'), " ", sum(g, dims=2), " ", sum(g * state'), " ", sum(x)))
    tuple(w' * g, g * x', sum(g, dims=2), g * state', (hw * state .+ w * x) ./ 2) # returning x crashes 2nd epoch
end
