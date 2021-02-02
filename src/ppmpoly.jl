function modelpoly!(
    ppm::AbstractPPM,
    seqs::Vector{Sequence};
    times::Vector{Vector{Float64}} = [Float64[]],
    train::Bool = true,
    predict::Bool = true,
    returndistribution::Bool = true,
    returnentropy::Bool = true
)
    nn = length(seqs)
    ns = map(length, seqs)

    # if all(isempty.(times)) && map(length, times) != ns
    #     error("times must either have length 0 or have lengths equal to those in seqs")
    # end
    # if length(ppm.alltime) > 0 && map(length, time) > 0 && time[1] < ppm.alltime[end]
    #     error("a sequence may not begin before the previous sequence finished")
    # end
    if map(y -> any(diff(y) .< 0), times) |> any
        error("decreasing values of time are not permitted")
    end

    if all(isempty.(times))
        times = [collect(1.0:n) for n in ns]
    end

    alltimes = vcat(times...) |> sort |> unique

    results = [
        SequencePrediction(returndistribution, returnentropy, ppm.decay)
        for i in 1:nn
    ]

    for t in alltimes
        timeevents = map(x->findfirst(isequal(t), x), times)

        # Predict
        for (i, e) in enumerate(timeevents)
            if isnothing(e); continue end
            x = seqs[i]
            j = findfirst(isequal(t), times[i])

            if predict
                context = if j == 1 || ppm.orderbound < 1
                    Sequence()
                else
                    subseq(x, max(1, j - ppm.orderbound), j - 1)
                end
                insertprediction!(results[i], predictsymbol(ppm, x[j], context, j, t))
            end
        end

        # Train
        for (i, e) in enumerate(timeevents)
            if isnothing(e); continue end
            x = seqs[i]
            j = findfirst(isequal(t), times[i])

            if train
                if ppm.decay; push!(ppm.alltime, t) end
                fullonly = false
                for h in max(1, j - ppm.orderbound):j
                    fullonly = ppminsert!(ppm, subseq(x, h, j), j, t, fullonly)
                end
                ppm.numobservations += 1
            end
        end
    end

    # return [resultdf(r) for r in results]
    return results
end
