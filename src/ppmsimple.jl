mutable struct RecordSimple <: AbstractRecord
    fullcount::Integer
    upexcount::Integer

    RecordSimple() = new(0, 0)
end


function add1!(a::RecordSimple, fullonly::Bool)
    if (a.fullcount >= typemax(typeof(a.fullcount)) ||
        a.upexcount >= typemax(typeof(a.upexcount)))
            error("cannot increment this record count any higher")
    end
    a.fullcount += 1
    if !fullonly
        a.upexcount += 1
    end
    return nothing
end


mutable struct PPMSimple <: AbstractPPM
    alphabetsize::Integer
    orderbound::Integer
    shortestdeterministic::Bool
    exclusion::Bool
    updateexclusion::Bool
    escape::String
    k::AbstractFloat
    decay::Bool
    subnfromm1dist::Bool
    lambdauseszeroweightsymbols::Bool
    debugsmooth::Bool

    numobservations::Integer
    alltime::Vector{Float64}

    data::Dict{Sequence, RecordSimple}

    function PPMSimple(
        _alphabetsize::Integer,
        _orderbound::Integer,
        _shortestdeterministic::Bool,
        _exclusion::Bool,
        _updateexclusion::Bool,
        _escape::String,
        _debugsmooth::Bool
    )
        if _alphabetsize <= 0
            error("alphabet size must be greater than 0")
        end
        new(
            _alphabetsize,
            _orderbound,
            _shortestdeterministic,
            _exclusion,
            _updateexclusion,
            _escape,
            getk(_escape),
            false,
            true,
            true,
            _debugsmooth,
            0,
            Float64[],
            Dict()
        )
    end
end


function ppminsert!(
    ppm::PPMSimple,
    x::Sequence,
    pos::Integer,
    time::AbstractFloat,
    fullonly::Bool
)
    if !haskey(ppm.data, x)
        record = RecordSimple()
        add1!(record, fullonly)
        ppm.data[x] = record
        return false
    else
        add1!(ppm.data[x], fullonly)
        return true
    end
end


function getlongestcontext(
    ppm::PPMSimple,
    context::Sequence,
    pos::Integer,
    time::AbstractFloat
)
    contextlen = length(context)
    upperbound = min(ppm.orderbound, contextlen)

    for order in upperbound:-1:0
        x = order == 0 ? Sequence() : subseq(context, (contextlen - order) + 1, contextlen)

        # Skip this iteration if the context doesn't exist in the tree
        # We don't store 0-grams in the tree
        if order > 0 && getweight(ppm, x, 0, 0.0, false) == 0.0
            continue
        end

        # Skip this iteration if we can't find a continuation for that context
        anycontinuation = false
        resize!(x, order + 1)
        for i in 1:ppm.alphabetsize
            x[order + 1] = i
            if getweight(ppm, x, 0, 0.0, false) > 0.0
                anycontinuation = true
                break
            end
        end
        if !anycontinuation
            continue
        end
        return order
    end
    return -1
end


function getweight(
    ppm::PPMSimple,
    ngram::Sequence,
    pos::Integer,
    time::AbstractFloat,
    updateexcluded::Bool
)
    return convert(AbstractFloat, getcount(ppm, ngram, updateexcluded))
end


function getcount(ppm::PPMSimple, x::Sequence, updateexcluded::Bool)
    if !haskey(ppm.data, x)
        return 0
    elseif updateexcluded
        return ppm.data[x].upexcount
    else
        return ppm.data[x].fullcount
    end
end


#=
The need to capture situations where the context_count is 0 is
introduced by Pearce (2005)'s decision to introduce exclusion
(see 6.2.3.3), though the thesis does not mention
this explicitly.
=#
function getlambda(
    ppm::PPMSimple,
    counts::Vector{Float64},
    contextcount::AbstractFloat,
    numdistictsymbols::Integer
)
    if ppm.debugsmooth
        println("calling ppm_simple.get_lambda()")
    end
    e = ppm.escape
    if contextcount <= 0.0
        return 0.0
    elseif e == "a"
        return lambdaa(ppm, counts, contextcount, numdistictsymbols)
    elseif e == "b"
        return lambdab(ppm, counts, contextcount, numdistictsymbols)
    elseif e == "c"
        return lambdac(ppm, counts, contextcount, numdistictsymbols)
    elseif e == "d"
        return lambdad(ppm, counts, contextcount, numdistictsymbols)
    elseif e == "ax"
        return lambdaax(ppm, counts, contextcount, numdistictsymbols)
    else
        error("unrecognised escape method")
    end
end


function aslist(ppm::PPMSimple)
    n = length(ppm.data)
    ngram = Vector{Sequence}(undef, n)
    fullcount = Vector{Int}(undef, n)
    upexcount = Vector{Int}(undef, n)

    i = 1
    for (k, v) in ppm.data
        ngram[i] = k
        fullcount = v.fullcount
        upexcount = v.upexcount
        i += 1
    end

    x = (ngram = ngram, fullcount = fullcount, upexcount = upexcount)
    return x
end
