Sequence = Vector{Integer}


function subseq(x::Sequence, first::Integer, last::Integer)
    if last > length(x) || last < first
        error("invalid subsequence indices")
    end
    n = 1 + last - first
    res = Sequence(undef, n)
    for j in 1:n
        res[j] = x[(j - 1) + first]
    end
    return res
end


function lastn(x::Sequence, n::Integer)
    if n < 0
        error("n cannot be less than 0")
    end
    originallength = length(x)
    if n > originallength
        error("cannot excise more elements than the sequence contains")
    end
    if n == 0
        return Sequence(undef, 0)
    else
        res = Sequence(undef, n)
        for i in 1:n
            res[i] = x[i + originallength - n]
        end
    end
    return res
end


function computeentropy(x::Vector{Float64})
    n = length(x)
    counter = 0.0
    for i in 1:n
        p = x[i]
        counter -= p * log2(p)
    end
    return counter
end


function normalisedistribution(x::Vector{Float64})
    total = 0.0
    n = length(x)
    distribution = copy(x)
    for i in 1:n
        total += distribution[i]
    end
    for i in 1:n
        distribution[i] = distribution[i] / total
    end
    return distribution
end
