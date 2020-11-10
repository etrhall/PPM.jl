
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
        posngrambegin = pos - ngramlength + 1
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
    pos = Vector{Vector{Integer}}(undef, n)
    time = Vector{Vector{Float64}}(undef, n)

    i = 1
    for (k, v) in ppm.data
        ngram[i] = k
        pos[i] = v.pos
        time[i] = ppm.alltime[pos[i]]
        i += 1
    end

    x = (ngram = ngram, pos = pos, time = time)
    return x
end
