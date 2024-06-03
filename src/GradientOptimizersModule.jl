module GradientOptimizersModule
    export GradientOptimizer, Adagrad, Descent

    abstract type GradientOptimizer end

    mutable struct Adagrad <: GradientOptimizer
        gradient_squared
        learning_rate
    end

    function (a::Adagrad)(g)
        a.gradient_squared = a.gradient_squared .+ g.^2
        delta = a.learning_rate .* g ./ (sqrt.(a.gradient_squared) .+ 10e-12)
        return delta
    end

    struct Descent <: GradientOptimizer
        learning_rate
    end

    function (d::Descent)(g)
        return d.learning_rate .* g
    end

    # WORK IN PROGRESS
    struct Adam <: GradientOptimizer
        momentum
        momentum_rate
        velocity
        velocity_rate
    end

    function (a::Adam)(g)
        momentum = a.momentum_rate .* a.momentum + (1 - a.momentum_rate) .* g
        velocity = a.velocity_rate * a.velocity + (1 - a.velocity_rate) .* (g.^2)
        a.momentum = momentum
        a.velocity = velocity
    end
end