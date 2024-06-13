module AccuracyModule
    using Statistics: mean

    function softmax(x)
        exp_x = exp.(x .- maximum(x, dims=1))
        return exp_x ./ sum(exp_x, dims=1)
    end

    function gradient(predictions, targets)
        probabilities = softmax(predictions)
        return Float32.(probabilities .- targets)
    end

    function loss(predictions, targets)
        probabilities = softmax(predictions)
        return -mean(sum(targets .* log.(probabilities), dims=1))
    end

    function one_cold(encoded)
        return [argmax(vec) for vec in eachcol(encoded)]
    end

    function accuracy(ŷ, y)
        pred_classes = one_cold(ŷ)
        true_classes = one_cold(y)
        return round(100 * mean(pred_classes .== true_classes); digits=2)
    end
end
