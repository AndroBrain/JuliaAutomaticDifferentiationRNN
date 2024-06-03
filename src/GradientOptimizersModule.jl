module GradientOptimizersModule
    export GradientOptimizer, Adagrad, Descent, RMSProp, Adam

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

    mutable struct RMSProp <: GradientOptimizer
       gradient_squared
       decay_rate
       learning_rate
    end

    function (r::RMSProp)(g)
       r.gradient_squared = r.gradient_squared .* r.decay_rate + g.^2 .* (1 - r.decay_rate)
       delta = r.learning_rate .* g ./ (sqrt.(r.gradient_squared) .+ 10e-12)
        return delta
    end

    struct Descent <: GradientOptimizer
        learning_rate
    end

    function (d::Descent)(g)
        return d.learning_rate .* g
    end

    mutable struct Adam <: GradientOptimizer
        gradient
        gradient_squared
        beta1
        beta2
        learning_rate
    end

    function (a::Adam)(g)
        a.gradient = a.gradient .* a.beta1 + g .* (1 - a.beta1)
        a.gradient_squared = a.gradient_squared * a.beta2 + g.^2 .* (1 - a.beta2)
        delta = a.learning_rate .* a.gradient ./ (sqrt.(a.gradient_squared) .+ 10e-12)
        return delta
    end
end