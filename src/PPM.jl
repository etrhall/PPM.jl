module PPM

export Sequence, PPMSimple, PPMDecay
export modelseq!

import Distributions: Distribution, Univariate, Continuous, Normal
import Random: AbstractRNG, MersenneTwister


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


abstract type AbstractRecord end


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


mutable struct RecordDecay <: AbstractRecord
    fullcount::Integer
    upexcount::Integer
    pos::Vector{Integer}

    RecordDecay() = new(0, 0, Integer[])
end


function insertrecord!(record::RecordDecay, pos::Integer, time::AbstractFloat)
    push!(record.pos, pos)
    return nothing
end


struct SymbolPrediction
    symbol::Integer
    pos::Integer
    time::AbstractFloat
    modelorder::Integer
    distribution::Vector{Float64}
    informationcontent::AbstractFloat

    function SymbolPrediction(
        _symbol::Integer,
        _pos::Integer,
        _time::AbstractFloat,
        _modelorder::Integer,
        _distribution::Vector{Float64}
    )
        if _symbol > length(_distribution)
            error("observed symbol not compatible with distribution dimensions")
        end
        new(
            _symbol,
            _pos,
            _time,
            _modelorder,
            _distribution,
            -log2(_distribution[_symbol])
        )
    end
end


struct SequencePrediction
    returndistribution::Bool
    returnentropy::Bool
    decay::Bool

    symbol::Vector{Integer}
    pos::Vector{Integer}
    time::Vector{Float64}
    modelorder::Vector{Integer}
    informationcontent::Vector{Float64}
    entropy::Vector{Float64}
    distribution::Vector{Vector{Float64}}

    function SequencePrediction(
        _returndistribution::Bool,
        _returnentropy::Bool,
        _decay::Bool
    )
        new(
            _returndistribution,
            _returnentropy,
            _decay,
            Int[],
            Int[],
            Float64[],
            Int[],
            Int[],
            Float64[],
            Vector[Float64[]]
        )
    end
end


function insertprediction!(a::SequencePrediction, x::SymbolPrediction)
    push!(a.symbol, x.symbol)
    push!(a.modelorder, x.modelorder)
    push!(a.informationcontent, x.informationcontent)
    if a.returnentropy
        push!(a.entropy, computeentropy(x.distribution))
    end
    if a.returndistribution
        push!(a.distribution, x.distribution)
    end
    if a.decay
        push!(a.pos, x.pos)
        push!(a.time, x.time)
    end
end


function aslist(a::SequencePrediction)
    x = Dict{String, Any}("symbol" => a.symbol)
    if a.decay
        push!(x, "pos" => a.pos)
        push!(x, "time" => a.time)
    end
    push!(x, "model_order" => a.modelorder)
    push!(x, "information_content" => a.informationcontent)
    if a.returnentropy
        push!(x, "entropy" => a.entropy)
    end
    if a.returndistribution
        push!(x, "distribution" => a.distribution)
    end
    return x
end


struct ModelOrder
    chosen::Integer
    longestavailable::Integer
    deterministicany::Bool
    deterministicshorted::Integer
    deterministicisselected::Bool
end


abstract type AbstractPPM end


mutable struct PPMModel <: AbstractPPM
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

    function PPMModel(
        _alphabetsize::Integer,
        _orderbound::Integer,
        _shortestdeterministic::Bool,
        _exclusion::Bool,
        _updateexclusion::Bool,
        _escape::String,
        _decay::Bool,
        _subnfromm1dist::Bool,
        _lambdauseszeroweightsymbols::Bool,
        _debugsmooth::Bool
    )
        if alphabetsize <= 0
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
            _decay,
            _subnfromm1dist,
            _lambdauseszeroweightsymbols,
            _debugsmooth,
            0,
            Float64[]
        )
    end
end


function ppminsert!(
    ppm::AbstractPPM,
    x::Sequence,
    pos::Integer,
    time::AbstractFloat,
    fullonly::Bool
)
    error("this shouldn't happen (1)")
    return true
end


function getweight(
    ppm::AbstractPPM,
    ngram::Sequence,
    pos::Integer,
    time::AbstractFloat,
    updateexcluded::Bool
)
    return 0.0
end


function getnumobservedsymbols(
    ppm::AbstractPPM,
    pos::Integer,
    time::AbstractFloat
)
    res = 0
    for i in 1:ppm.alphabetsize
        symbol = Sequence([i])
        weight = getweight(ppm, symbol, pos, time, false)
        if weight > 0.0
            res += 1
        end
    end
    return res
end


function getcontextcount(
    ppm::AbstractPPM,
    counts::Vector{Float64},
    excluded::Vector{Bool}
)
    contextcount = 0.0
    for i in 1:ppm.alphabetsize
        if !excluded[i]
            contextcount += counts[i]
        end
    end
    return contextcount
end


function modelseq!(
    ppm::AbstractPPM,
    x::Sequence,
    time::Vector{Float64} = Float64[],
    train::Bool = true,
    predict::Bool = true,
    returndistribution::Bool = true,
    returnentropy::Bool = true
)
    n = length(x)
    if ppm.decay && n != length(time)
        error("time must either have length 0 or have length equal to x")
    end
    if length(ppm.alltime) > 0 && length(time) > 0 && time[1] < ppm.alltime[end]
        error("a sequence may not begin before the previous sequence finished")
    end

    result = SequencePrediction(returndistribution, returnentropy, ppm.decay)

    for i in 1:n
        posi = ppm.numobservations + 1  # pos changed to be 1-indexed
        timei = ppm.decay ? time[i] : 0.0
        # Predict
        if predict
            context = if i == 1 || ppm.orderbound < 1
                Sequence()
            else
                subseq(x, max(1, i - ppm.orderbound), i - 1)
            end
            insertprediction!(result, predictsymbol(ppm, x[i], context, posi, timei))
        end
        # Train
        if train
            ppm.decay ? push!(ppm.alltime, timei) : nothing
            fullonly = false
            for h in max(1, i - ppm.orderbound):i
                fullonly = ppminsert!(ppm, subseq(x, h, i), posi, timei, fullonly)
            end
            ppm.numobservations += 1
        end
    end
    return aslist(result)
end


function predictsymbol(
    ppm::AbstractPPM,
    symbol::Integer,
    context::Sequence,
    pos::Integer,
    time::AbstractFloat
)
    if symbol < 0
        error("symbols must be greater than or equal to 0")
    end
    if symbol > ppm.alphabetsize
        error("symbols cannot exceed (alphabet_size - 1)")
    end
    modelorder = getmodelorder(ppm, context, pos, time)
    dist = getprobabilitydistribution(ppm, context, modelorder, pos, time)

    out = SymbolPrediction(symbol, pos, time, modelorder.chosen, dist)
    return out
end


function getprobabilitydistribution(
    ppm::AbstractPPM,
    context::Sequence,
    modelorder::ModelOrder,
    pos::Integer,
    time::AbstractFloat
)
    excluded = fill(false, ppm.alphabetsize)
    dist = getsmootheddistribution(
        ppm,
        context,
        modelorder,
        modelorder.chosen,
        pos,
        time,
        excluded
    )
    return normalisedistribution(dist)
end


function getsmootheddistribution(
    ppm::AbstractPPM,
    context::Sequence,
    modelorder::ModelOrder,
    order::Integer,
    pos::Integer,
    time::AbstractFloat,
    excluded::Vector{Bool}
)
    if order == -1
        return getorderminus1distribution(ppm, pos, time)
    else
        updateexcluded = ppm.updateexclusion
        if (order == modelorder.chosen &&
            ppm.shortestdeterministic &&
            ppm.updateexclusion &&
            modelorder.deterministicisselected)
                updateexcluded = false
        end

        ngram = lastn(context, order)
        resize!(ngram, order + 1)

        counts = zeros(Float64, ppm.alphabetsize)
        numdistinctsymbols = 0
        predicted = zeros(Bool, ppm.alphabetsize)

        for i in 1:ppm.alphabetsize
            ngram[order + 1] = i
            counts[i] = getweight(ppm, ngram, pos, time, updateexcluded)
            if counts[i] > 0.0
                predicted[i] = true
                numdistinctsymbols += 1
            else
                predicted[i] = false
            end
            counts[i] = modifycount(ppm, counts[i])
        end

        contextcount = getcontextcount(ppm, counts, excluded)
        lambda = getlambda(ppm, counts, contextcount, numdistinctsymbols)

        alphas = getalphas(ppm, lambda, counts, contextcount)

        if ppm.debugsmooth
            println()
            println("*** order = $order ***")
            println("pos = ", pos)
            println("time = ", time)
            println("modelorder.chosen = ", modelorder.chosen)
            println("ppm.shortestdeterministic = ", ppm.shortestdeterministic)
            println("ppm.updateexclusion = ", ppm.updateexclusion)
            print("modelorder.deterministicisselected = ")
            print(modelorder.deterministicisselected)
            println("context = ", lastn(context, order))
            println("updateexcluded = ", updateexcluded)
            println("counts = ", counts)
            println("contextcounts = ", contextcount)
            println("lambda = ", lambda)
            println("alphas = ", alphas)
        end

        if ppm.exclusion
            for i in 1:ppm.alphabetsize
                #=
                There is a choice here:
                do we exclude symbols that have alphas greater than 0
                i.e. their counts survive addition of k),
                or do we exclude any symbol that is present in the tree at all,
                even if adding k takes it down to 0?

                Since decay-based models don't have exclusion,
                we only have to think about normal PPM models.
                All of these models apart from PPM-B have k > -1,
                in which case there is no difference between the strategies.
                We only have to worry for PPM-B.

                Following Bunton (1996) and Pearce (2005)'s implementation,
                we adopt the latter strategy, excluding symbols even
                if their alphas are equal to 0, as long as they were present
                in the tree.
                =#
                if predicted[i]
                    excluded[i] = true
                end
            end
            if ppm.debugsmooth
                println("new excluded = ", excluded)
            end
        end

        lowerorderdistribution = getsmootheddistribution(
            ppm, context, modelorder, order - 1, pos, time, excluded
        )

        res = zeros(Float64, ppm.alphabetsize)
        for i in 1:ppm.alphabetsize
            res[i] = alphas[i] + (1 - lambda) * lowerorderdistribution[i]
        end

        if ppm.debugsmooth
            println("order ", order)
            println("probability distribution = ", res)
        end

        return res
    end
end


function getalphas(
    ppm::AbstractPPM,
    lambda::AbstractFloat,
    counts::Vector{Float64},
    contextcount::AbstractFloat
)
    if lambda > 0
        res = zeros(Float64, ppm.alphabetsize)
        for i in 1:ppm.alphabetsize
            res[i] = lambda * counts[i] / contextcount
        end
        return res
    else
        res = zeros(Float64, ppm.alphabetsize)
        return res
    end

end


#=
The need to capture situations where the contextcount is 0 is
introduced by Pearce (2005)'s decision to introduce exclusion
(see 6.2.3.3), though the thesis does not mention
this explicitly.
=#
function getlambda(
    ppm::AbstractPPM,
    counts::Vector{Float64},
    contextcount::AbstractFloat,
    numdistinctsymbols::Integer
)
    error("this virtual get_lambda method should never be called directly")
    return 0.0
end


function getk(e::String)
    if e == "a"
        return 0.0
    elseif e == "b"
        return -1.0
    elseif e == "c"
        return 0.0
    elseif e == "d"
        return -0.5
    elseif e == "ax"
        return 0.0
    else
        error("unrecognised escape method")
    end
end


function geteffectivedistinctsymbols(
    ppm::AbstractPPM,
    numdistinctsymbols::Integer,
    counts::Vector{Float64}
)
    if ppm.lambdauseszeroweightsymbols
        return numdistinctsymbols
    else
        return countpositivevalues(counts)
    end
end


function lambdaa(
    ppm::AbstractPPM,
    counts::Vector{Float64},
    contextcount::AbstractFloat,
    numdistinctsymbols::Integer
)
    if ppm.debugsmooth
        println("lambda_a, context_count = ", contextcount)
    end
    return contextcount / (contextcount + 1.0)
end


function lambdab(
    ppm::AbstractPPM,
    counts::Vector{Float64},
    contextcount::AbstractFloat,
    numdistinctsymbols::Integer
)
    effectivedistinctsymbols = geteffectivedistinctsymbols(ppm, numdistinctsymbols, counts)

    return contextcount / (contextcount + effectivedistinctsymbols)
end


function lambdac(
    ppm::AbstractPPM,
    counts::Vector{Float64},
    contextcount::AbstractFloat,
    numdistinctsymbols::Integer
)
    effectivedistinctsymbols = geteffectivedistinctsymbols(ppm, numdistinctsymbols, counts)

    return contextcount / (contextcount + effectivedistinctsymbols)
end


function lambdad(
    ppm::AbstractPPM,
    counts::Vector{Float64},
    contextcount::AbstractFloat,
    numdistinctsymbols::Integer
)
    effectivedistinctsymbols = geteffectivedistinctsymbols(ppm, numdistinctsymbols, counts)

    return contextcount / (contextcount + effectivedistinctsymbols / 2.0)
end


function lambdaax(
    ppm::AbstractPPM,
    counts::Vector{Float64},
    contextcount::AbstractFloat,
    numdistinctsymbols::Integer
)
    #=
    Note - there is a mistake in the reference papers,
    Pearce & Wiggins (2004), also Pearce (2005);
    the 1.0 is missing from the equation.
    Our version is consistent with the context literature though,
    and consistent with Pearce's LISP implementation.

    We generalise the definition of singletons to decayed counts between
    0 and 1. This is a bit hacky though, and the escape method
    should ultimately be reconfigured for new decay functions.
    =#
    return contextcount / (contextcount + numsingletons(counts) + 1.0)
end


function numsingletons(x::Vector{Float64})
    n = length(x)
    res = 0
    for i in 1:n
        if x[i] > 0 && x[i] <= 1
            res += 1
        end
    end
    return res
end


function modifycount(ppm::AbstractPPM, count::AbstractFloat)
    if ppm.k == 0.0 || count == 0.0
        return count
    else
        x = count + ppm.k
        if x > 0.0
            return x
        else
            return 0.0
        end
    end
end


function countpositivevalues(x::Vector{Float64})
    n = length(x)
    res = 0
    for i in 1:n
        if x[i] > 0
            res += 1
        end
    end
    return res
end


function getorderminus1distribution(
    ppm::AbstractPPM,
    pos::Integer,
    time::AbstractFloat
)
    #=
    See Bunton (1996, p. 82): alpha(s0) comes from the 3-arg version of count(),
    which does not include exclusion or subtraction
    of the k parameter (see escape method).
    It instead corresponds to the number of symbols that the model
    has ever seen.
    =#

    denominator = ppm.alphabetsize + 1

    if ppm.subnfromm1dist
        # This is disabled for decay-based models
        numobservedsymbols = getnumobservedsymbols(ppm, pos, time)
        denominator -= numobservedsymbols
    end

    p = 1.0 / denominator
    res = fill(p, ppm.alphabetsize)

    if ppm.debugsmooth
        println("order minus 1 distribution = ", res)
    end
    return res
end


function getmodelorder(
    ppm::AbstractPPM,
    context::Sequence,
    pos::Integer,
    time::AbstractFloat
)
    longestavailable = getlongestcontext(ppm, context, pos, time)
    chosen = copy(longestavailable)

    detshortest = -1
    detany = false
    detisselected = false

    if ppm.shortestdeterministic
        detshortest = getshortestdeterministiccontext(ppm, context, pos, time)
        detany = detshortest >= 0
        if detany
            if detshortest < longestavailable
                detisselected = true
                chosen = detshortest
            end
        end
    end
    return ModelOrder(chosen, longestavailable, detany, detshortest, detisselected)
end


function getlongestcontext(
    ppm::AbstractPPM,
    context::Sequence,
    pos::Integer,
    time::AbstractFloat
)
    error("this shouldn't happen (2)")
    return 0
end


function getshortestdeterministiccontext(
    ppm::AbstractPPM,
    context::Sequence,
    pos::Integer,
    time::AbstractFloat
)
    len = length(context)
    res = -1
    for order in 0:min(len, ppm.orderbound)
        effectivecontext = if order == 0
            Sequence()
        else
            subseq(context, (len - order) + 1, len)
        end
        if isdeterministiccontext(ppm, effectivecontext, pos, time)
            res = order
            break
        end
    end
    return res
end


function isdeterministiccontext(
    ppm::AbstractPPM,
    context::Sequence,
    pos::Integer,
    time::AbstractFloat
)
    numcontinuations = 0
    for i in 1:ppm.alphabetsize
        ngram = copy(context)
        push!(ngram, i)
        weight = getweight(ppm, ngram, pos, time, false)  # update exclusion
        if weight > 0
            numcontinuations += 1
            if numcontinuations > 1
                break
            end
        end
    end
    return numcontinuations == 1
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


mutable struct PPMDecay <: AbstractPPM
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

    data::Dict{Sequence, RecordDecay}

    bufferlengthtime::AbstractFloat
    bufferlengthitems::Integer
    bufferweight::AbstractFloat
    onlylearnfrombuffer::Bool
    onlypredictfrombuffer::Bool
    stmweight::AbstractFloat
    stmduration::AbstractFloat
    stmhalflife::AbstractFloat # computed from stmweight, ltmweight, and stmduration
    ltmweight::AbstractFloat
    ltmhalflife::AbstractFloat
    ltmasymptote::AbstractFloat
    noise::AbstractFloat
    noisemean::AbstractFloat
    disablenoise::Bool
    seed::Integer
    debugdecay::Bool

    randomengine::AbstractRNG
    noisegenerator::Distribution{Univariate,Continuous}

    function PPMDecay(
        _alphabetsize::Integer,
        _orderbound::Integer,
        _decaypar::NamedTuple,
        _seed::Integer,
        _debugsmooth::Bool,
        _debugdecay::Bool
    )
        if _alphabetsize <= 0
            error("alphabet size must be greater than 0")
        end

        _stmhalflife = (
            (log(2.0) * _decaypar.stmduration) /
            (log(_decaypar.stmweight / _decaypar.ltmweight))
        )

        if _decaypar.ltmweight > _decaypar.stmweight
            error("ltmweight cannot be greater than stmweight")
        end

        if _decaypar.ltmweight <= 0.0
            error("ltmweight must be positive")
        end

        if _decaypar.stmweight <= 0.0
            error("stmweight must be positive")
        end

        if _decaypar.stmduration < 0
            error("stmduration cannot be negative")
        end

        if _decaypar.ltmhalflife <= 0
            error("ltmhalflife must be positive")
        end

        if _decaypar.ltmasymptote < 0
            error("ltmasymptote must be non-negative")
        end

        if _decaypar.ltmasymptote > _decaypar.ltmweight
            error("ltmasymptote cannot be greater than ltmweight")
        end

        # if _escape != "a"
        #     error("escape method must be 'a' for decay-based models")
        # end

        if (_decaypar.onlylearnfrombuffer
            && _decaypar.bufferlengthitems - 1 < _orderbound)
                error(
                    "if onlylearnfrombuffer is TRUE,
                    order bound cannot be greater than bufferlengthitems - 1"
                )
        end

        _noisemean = _decaypar.noise * sqrt(2.0 / π) # mean of abs(normal distribution)

        _randomengine = MersenneTwister(_seed)
        _noisegenerator = Normal(0.0, _decaypar.noise)

        new(
            _alphabetsize,
            _orderbound,
            false,  # shortestdeterministic
            false,  # exclusion
            false,  # updateexclusion
            "a",    # escape
            getk("a"),
            true,   # decay
            false,  # subnfromm1dist
            false,  # lambdauseszeroweightsymbols
            _debugsmooth,
            0,
            Float64[],
            Dict(),
            _decaypar.bufferlengthtime,
            _decaypar.bufferlengthitems,
            _decaypar.bufferweight,
            _decaypar.onlylearnfrombuffer,
            _decaypar.onlypredictfrombuffer,
            _decaypar.stmweight,
            _decaypar.stmduration,
            _stmhalflife,
            _decaypar.ltmweight,
            _decaypar.ltmhalflife,
            _decaypar.ltmasymptote,
            _decaypar.noise,
            _noisemean,
            false,
            _seed,
            _debugdecay,
            _randomengine,
            _noisegenerator
        )
    end
end


function ppminsert!(
    ppm::PPMDecay,
    x::Sequence,
    pos::Integer,
    time::AbstractFloat,
    fullonly::Bool
)
    if ppm.onlylearnfrombuffer
        #=
        We skip the n-gram insertion if we find that the n-gram
        doesn't fit in the buffer.
        Note that we only have to check the temporal constraint;
        the positional constraint was checked when the
        model's order bound was originally specified.
        =#
        ngramlength = length(x)
        posngrambegin = pos - ngramlength
        timengrambegin = ppm.alltime[posngrambegin]
        if time - timengrambegin >= ppm.bufferlengthtime
            return true
        end
    end
    if !haskey(ppm.data, x)
        record = RecordDecay()
        insertrecord!(record, pos, time)
        ppm.data[x] = record
        return false
    else
        insertrecord!(ppm.data[x], pos, time)
        return true
    end
end


function getlongestcontext(
    ppm::PPMDecay,
    context::Sequence,
    pos::Integer,
    time::AbstractFloat
)
    maxcontextsize = length(context)
    if maxcontextsize > ppm.orderbound
        error("this shouldn't happen (3)")
    end
    if ppm.onlypredictfrombuffer
        for contextsize in maxcontextsize:-1:1
            poscontextbegin = pos - contextsize
            timecontextbegin = ppm.alltime[poscontextbegin]
            if time - timecontextbegin <= ppm.bufferlengthtime
                return contextsize
            end
        end
        return 0
    else
        return maxcontextsize
    end
end


function getweight(
    ppm::PPMDecay,
    ngram::Sequence,
    pos::Integer,
    time::AbstractFloat,
    updateexcluded::Bool
)
    data = getrecord(ppm, ngram)

    N = length(ppm.alltime)
    n = length(data.pos)

    weight = 0.0
    for i in 1:n
        if ppm.debugdecay
            println()
            println("observing from pos = $pos, time = $time")
            println("memory $i / $n")
        end

        if data.pos[i] > pos
            error("tried to predict using training data from the future")
        end
        if data.pos[i] < 1
            error("data.pos cannot be less than 1")
        end

        # # Original buffer version
        # positembufferfails = data.pos[i] + ppm.bufferlengthitems
        # temporalbufferfailtime = data.time[i] + ppm.bufferlengthtime

        positembufferfails = data.pos[i] + max(0, ppm.bufferlengthitems - length(ngram) + 1)
        if ppm.debugdecay
            println("positembufferfails = ", positembufferfails)
        end

        itembufferfailed = positembufferfails <= N  # <= pos
        if ppm.debugdecay
            println("itembufferfailed = ", itembufferfailed)
        end

        posngrambegan = data.pos[i] - length(ngram) + 1

        temporalbufferfailtime = ppm.alltime[posngrambegan] + ppm.bufferlengthtime
        if ppm.debugdecay
            println("temporalbufferfailtime = ", temporalbufferfailtime)
        end

        bufferfailtime = NaN

        if itembufferfailed
            timewhenitembufferfailed = ppm.alltime[positembufferfails]
            if ppm.debugdecay
                println("timewhenitembufferfailed = ", timewhenitembufferfailed)
            end
            bufferfailtime = min(timewhenitembufferfailed, temporalbufferfailtime)
        else
            bufferfailtime = temporalbufferfailtime
        end

        timesincebufferfail = time - bufferfailtime

        if ppm.debugdecay
            println("bufferfailtime = ", bufferfailtime)
            println("timesincebufferfail = ", timesincebufferfail)
        end

        weightincrement = NaN

        if timesincebufferfail < 0
            if ppm.debugdecay
                println("buffer didn't fail")
            end
            weightincrement = ppm.bufferweight
        else
            if ppm.debugdecay
                println("buffer failed")
            end
            weightincrement = decaystmltm(ppm, timesincebufferfail)
        end

        if ppm.debugdecay
            println("weightincrement = ", weightincrement)
        end

        weight += weightincrement
    end

    if !ppm.disablenoise
        noise = abs(rand(ppm.randomengine, ppm.noisegenerator))
        weight += noise
    end

    if ppm.debugdecay
        println("total weight = ", weight)
    end

    return weight
end


function decaystmltm(ppm::PPMDecay, elapsedtime::AbstractFloat)
    stm = elapsedtime < ppm.stmduration
    res = NaN
    if stm
        res = decayexp(
            ppm.stmweight,
            elapsedtime,
            ppm.stmhalflife,
            0.0
        )
    else
        res = decayexp(
            ppm.ltmweight,
            elapsedtime - ppm.stmduration,
            ppm.ltmhalflife,
            ppm.ltmasymptote
        )
    end
    return res
end


function decayexp(
    start::AbstractFloat,
    elapsedtime::AbstractFloat,
    halflife::AbstractFloat,
    asymptote::AbstractFloat
)
    if halflife <= 0.0
        error("half life must be positive")
    end
    return asymptote + (start - asymptote) * 2.0^-(elapsedtime / halflife)
end


function getlambda(
    ppm::PPMDecay,
    counts::Vector{Float64},
    contextcount::AbstractFloat,
    numdistinctsymbols::Integer
)
    if ppm.debugdecay
        println("calling PPMDecay getlambda()")
    end
    totalexpectednoise = NaN
    if ppm.disablenoise
        totalexpectednoise = 0.0
    else
        totalexpectednoise = ppm.noisemean * ppm.alphabetsize
    end
    adjcontextcount = max(contextcount - totalexpectednoise, 0.0)

    if contextcount <= 0.0
        if ppm.debugsmooth
            # This should actually be adjcontextcount, but does it matter?
            # no, because adjcontextcount always <= contextcount
            println("contextcount <= 0.0 so lambda = 0.0")
        end
        return 0.0
    else
        if ppm.debugsmooth
            println("calling lambdaa...")
        end
        return lambdaa(ppm, counts, adjcontextcount, -99)  # last parameter ignored
    end
end


function getrecord(ppm::PPMDecay, x::Sequence)
    if !haskey(ppm.data, x)
        blank = RecordDecay()
        return blank
    else
        return ppm.data[x]
    end
end


function aslist(ppm::PPMDecay)
    n = length(ppm.data)
    ngram = Vector{Sequence}(undef, n)
    pos = Vector{Integer}(undef, n)
    time = Vector{Float64}(undef, n)

    i = 1
    for (k, v) in data
        ngram[i] = k
        pos[i] = v.pos
        time[i] = ppm.alltime[pos[i]]
        i += 1
    end

    x = (ngram = ngram, pos = pos, time = time)
    return x
end


end
