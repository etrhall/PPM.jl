Sequence = Vector{Integer}


function getalphabet(x)
    k = map(Symbol, unique(x)) |> sort
    n = length(k)
    return (; zip(k, 1:n)...)
end


function assequence(x::Union{Vector, AbstractString})
    n = length(x)
    levels = unique(x)
    seq = Sequence(undef, n)
    for i in 1:n
        seq[i] = findfirst(isequal(x[i]), levels)
    end
    return seq
end


function assequence(x::Vector, alphabet::NamedTuple)
    return map(y->alphabet[Symbol(y)], x) |> Sequence
end


function assequence(x::AbstractString, alphabet::NamedTuple)
    return map(y->alphabet[Symbol(y)], split(x, "")) |> Sequence
end


function subseq(x::Sequence, first::Integer, last::Integer)
    if last > length(x) || last < first
        error("invalid subsequence indices")
    end
    return x[first:last]
end


function lastn(x::Sequence, n::Integer)
    if n < 0
        error("n cannot be less than 0")
    end
    if n > length(x)
        error("cannot excise more elements than the sequence contains")
    end
    return x[(end - n + 1):end]
end


computeentropy(x::Vector{Float64}) = sum(p->(-p * log2(p)), x)


normalisedistribution(x::Vector{Float64}) = map(y->(y / sum(x)), x)


# function resultdf(result, display::Bool = true)
#     DataFrame(
#         symbol = result.symbol,
#         # position = result.pos,
#         # time = result.time,
#         model_order = result.modelorder,
#         information_content = result.informationcontent,
#         entropy = result.entropy
#     )
#     # pretty_table(
#     #     df,
#     #     ["symbol", "order", "information content", "entropy"],
#     #     alignment = :r,
#     #     formatters = ft_printf("%5.4f", [5, 6]),
#     #     border_crayon = crayon"blue",
#     #     header_crayon = crayon"bold red"
#     # )
# end
