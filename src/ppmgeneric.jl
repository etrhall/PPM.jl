abstract type AbstractRecord end

abstract type AbstractPPM end


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
            Vector{Vector{Float64}}()
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
    push!(x, "modelorder" => a.modelorder)
    push!(x, "informationcontent" => a.informationcontent)
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


mutable struct PPMGeneric <: AbstractPPM
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

    function PPMGeneric(
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


"""Gets the weight (or count) of a given n-gram in a trained PPM moel.

`ppm`:
A PPM model object as produced by (for example)
``newppmsimple`` or ``newppmdecay``.

`ngram`:
An integer vector defining the n-gram to be queried.

`pos`:
The nominal 'position' at which the n-gram is retrieved
(only relevant for decay-based models).

`time`:
The nominal 'time' at which the n-gram is retrieved
(only relevant for decay-based models).

`update_excluded`:
Whether to retrieve update-excluded counts or not.

Returns a numeric scalar identifying the weight of the n-gram.
"""
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


"""
modelseq!(
    `ppm`::AbstractPPM,
    `x`::Sequence,
    `time`::Vector{Float64} = ``Float64[]``,
    `train`::Bool = ``true``,
    `predict`::Bool = ``true``,
    `returndistribution`::Bool = ``true``,
    `returnentropy`::Bool = ``true``
)


Analyses a sequence using a PPM model.

`ppm`:
A PPM model object as produced by (for example)
```newppmsimple``` or ```newppmdecay```.

`x`:
An integer vector defining the input sequence
(equivalently a numeric vector containing solely integers,
or a factor vector, both of which which will be coerced to integer vectors).

`time`:
Timepoints corresponding to each element of the argument `x`.
Only used by certain model types (e.g. decay-based models).

`train`:
Whether or not the model should learn from the incoming sequence.

`predict`:
Whether or not to generate predictions for each element of
the incoming sequence.

`returndistribution`:
Whether or not to return the conditional distribution over each
potential continuation as part of the model output
(ignored if predict = ``false``).

`returnentropy`:
Whether or not to return the entropy of each event prediction
(ignored if predict = ``false``).

@return
A data frame which will be empty if predict = ``false``
and otherwise will contain one row for each element in the sequence,
with the following keys:

- *symbol* - the symbol being predicted. This should be identical
to the input argument `x`.
- *modelorder* - the model order used for generating predictions.
- *informationcontent* - the information content
(i.e. negative log probability, base 2) of the observed symbol.
- *entropy* - the expected information content when
predicting the symbol.
- *distribution* - the predictive probability distribution for the
symbol, conditioned on the preceding symbols.
"""
function modelseq!(
    ppm::AbstractPPM,
    x::Sequence;
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
    if any(diff(time) .< 0)
        error("decreasing values of time are not permitted")
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
