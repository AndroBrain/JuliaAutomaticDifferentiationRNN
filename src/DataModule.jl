module DataModule
    using MLDatasets, Random

    function preprocess(split::Symbol; one_hot::Bool=true)
        features, targets = load(split)
        x1dim = reshape(features, 28 * 28, :)
        yhot = one_hot ? hot_encode(targets, 0:9) : targets

        perm = randperm(size(x1dim, 2))
        x1dim = x1dim[:, perm]
        yhot = yhot[:, perm]
        return x1dim, yhot
    end

    function load(split::Symbol)
        data = MLDatasets.MNIST(split = split)
        return data.features, data.targets
    end

    function hot_encode(targets, classes)
        one_hot = zeros(Int, length(classes), length(targets))
        for (i, class) in enumerate(classes)
            one_hot[i, findall(x -> x == class, targets)] .= 1
        end
        return one_hot
    end

    function batch(data, n)
        arr = Array{Matrix{Float32}}(undef, 0)
        for i in 1:size(data,2) / n
            i = floor(Int, i)
            push!(arr, data[:, (i * n - n + 1):i*n])
        end
        return arr
    end

    function mat_shuffle(x, y)
        perm = randperm(size(x, 2))
        x = x[:, perm]
        y = y[:, perm]
        return x,y
    end
end
