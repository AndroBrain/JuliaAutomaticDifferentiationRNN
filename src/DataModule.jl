module DataModule
    using MLDatasets, Random

    function preprocess(split::Symbol; one_hot::Bool=true)
        features, targets = load(split)
        x1dim = reshape(features, 28 * 28, :)
        yhot = one_hot ? hot_encode(targets, 0:9) : targets
        return x1dim, yhot
    end

    function load(split::Symbol)
        data = MLDatasets.MNIST(split=split)
        return data.features, data.targets
    end

    function hot_encode(targets, classes)
        one_hot = zeros(Int, length(classes), length(targets))
        for (i, class) in enumerate(classes)
            one_hot[i, findall(x -> x == class, targets)] .= 1
        end
        return one_hot
    end
end
