module GradientOptimizersModule
    export GradientOptimizer, Adagrad, Descent

    abstract type GradientOptimizer end

    struct Adagrad <: GradientOptimizer
        learning_rate
        small_value
        step
    end

    function (a::Adagrad)(g)
        s = a.step .+ g.*g
        return a.learning_rate .* g ./ (sqrt.(s) .+ a.small_value)
    end

    struct Descent <: GradientOptimizer
        learning_rate
    end

    function (d::Descent)(g)
        return d.learning_rate .* g
    end
end