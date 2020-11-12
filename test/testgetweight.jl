@testset "get weight" begin
    @testset "misc" begin
        function f(
            seq::Sequence,
            ngram::Sequence,
            pos::Integer,
            time::AbstractFloat,
            datatime::Vector{Float64};
            alphabetsize::Integer = 100,
            noise::AbstractFloat = 0.0,
            orderbound::Integer = 3,
            args...
        )
            mod = newppmdecay(
                alphabetsize;
                noise = noise,
                orderbound = orderbound,
                args...
            )
            modelseq!(
                mod,
                seq;
                time = datatime,
                train = true,
                predict = false
            )
            return getweight(mod, ngram, pos, time, false)
        end

        function decayexp(
            timeelapsed::AbstractFloat,
            halflife::AbstractFloat,
            starttime::AbstractFloat,
            endtime::AbstractFloat
        )
            lambda = log(2) / halflife
            return endtime + (starttime - endtime) * exp(-lambda * timeelapsed)
        end

        ## Item buffers
        # Buffer = 10 - everything at full stmrate
        res = f(
            fill(1, 9) |> Sequence,
            Sequence([1]),
            10,
            10.0,
            collect(1.0:9.0);
            noise = 0.0,
            bufferlengthtime = 999999.0,
            bufferlengthitems = 10,
            bufferweight = 1.0,
            stmduration = 0.0,
            ltmhalflife = 0.000000001
        )
        @test res ≈ 9.0

        # No more than 10 cases can be counted
        res = f(
            fill(1, 15) |> Sequence,
            Sequence([1]),
            16,
            16.0,
            collect(1.0:15.0);
            noise = 0.0,
            bufferlengthtime = 999999.0,
            bufferlengthitems = 10,
            bufferweight = 1.0,
            stmduration = 0.0,
            ltmhalflife = 0.000000001
        )
        @test res ≈ 10.0

        # Now set a non-zero ltm_weight
        res = f(
            fill(1, 15) |> Sequence,
            Sequence([1]),
            16,
            16.0,
            collect(1.0:15.0);
            noise = 0.0,
            bufferlengthtime = 999999.0,
            bufferlengthitems = 10,
            bufferweight = 1.0,
            stmduration = 0.0,
            ltmweight = 0.1,
            ltmhalflife = 1e60
        )
        @test res ≈ 10.0 + 0.5

        # Now to distinguish time from position,
        # we need to set a non-zero half-life.

        # Nothing within the buffer decays
        res = f(
            fill(1, 10) |> Sequence,
            Sequence([1]),
            10,
            10.0,
            collect(1.0:10.0);
            noise = 0.0,
            bufferlengthtime = 999999.0,
            bufferlengthitems = 10,
            bufferweight = 1.0,
            stmduration = 0.0,
            ltmweight = 1.0,
            # ltmweight = 0.0,
            ltmhalflife = 1.0
        )
        @test res ≈ 10.0

        # Past the buffer, we decay with a half-life of 1
        res = f(
            fill(1, 11) |> Sequence,
            Sequence([1]),
            12,
            12.0,
            collect(1.0:11.0);
            noise = 0.0,
            bufferlengthtime = 999999.0,
            bufferlengthitems = 10,
            bufferweight = 1.0,
            stmduration = 0.0,
            ltmhalflife = 1.0,
            ltmweight = 1.0
        )
        @test res ≈ 10.0 + 0.5

        res = f(
            fill(1, 11) |> Sequence,
            Sequence([1]),
            12,
            13.0,
            collect(1.0:11.0);
            noise = 0.0,
            bufferlengthtime = 999999.0,
            bufferlengthitems = 10,
            bufferweight = 1.0,
            stmduration = 0.0,
            ltmhalflife = 1.0
        )
        @test res ≈ 10.0 + 0.25

        # Time buffers
        res = f(
            fill(1, 10) |> Sequence,
            Sequence([1]),
            10,
            10.0,
            collect(1.0:0.5:5.5);
            noise = 0.0,
            bufferlengthtime = 7.0,
            bufferlengthitems = 1000,
            bufferweight = 1.0,
            stmduration = 0.0,
            ltmhalflife = 1.0
        )
        ans = (
            decayexp(2.0, 1.0, 1.0, 0.0) +
            decayexp(1.5, 1.0, 1.0, 0.0) +
            decayexp(1.0, 1.0, 1.0, 0.0) +
            decayexp(0.5, 1.0, 1.0, 0.0) +
            6.0
        )
        @test res ≈ ans

        ## Buffers with longer n-grams

        # With a buffer of length 4,
        # an n-gram of length 2 with its final symbol at pos = 1
        # should still be in the buffer two symbols later (pos = 3)
        # and quit it at pos = 4.

        res = f(
            collect(1:4) |> Sequence,
            Sequence([1, 2]),
            3,
            3.0,
            collect(0.0:3.0);
            noise = 0.0,
            bufferlengthtime = 999999.0,
            bufferlengthitems = 4,
            bufferweight = 1.0,
            stmduration = 0.0,
            ltmhalflife = 1.0,
            ltmweight = 0.1
        )
        @test res ≈ 1.0

        res = f(
            collect(1:5) |> Sequence,  # <------
            Sequence([1, 2]),
            4,  # <------
            4.0,  # <------
            collect(0.0:4.0);  # <------
            noise = 0.0,
            bufferlengthtime = 999999.0,
            bufferlengthitems = 4,
            bufferweight = 1.0,
            stmduration = 0.0,
            ltmhalflife = 1.0,
            ltmweight = 0.1
        )
        @test res ≈ 0.1

        # With a buffer of time length 4,
        # an n-gram of length 2 with its first symbol at pos/time = 1
        # should still be in the buffer at time = 4
        # and quit it at time = 5

        res = f(
            collect(1:6) |> Sequence,
            Sequence([2, 3]),
            4,
            4.0,
            collect(0.0:5.0);
            noise = 0.0,
            bufferlengthtime = 4.0,
            bufferlengthitems = 1000,
            bufferweight = 1.0,
            stmduration = 0.0,
            ltmhalflife = 1.0,
            ltmweight = 0.1
        )
        @test res ≈ 1.0

        res = f(
            collect(1:6) |> Sequence,
            Sequence([2, 3]),
            5,
            5.0,
            collect(0.0:5.0);
            noise = 0.0,
            bufferlengthtime = 4.0,
            bufferlengthitems = 1000,
            bufferweight = 1.0,
            stmduration = 0.0,
            ltmhalflife = 1.0,
            ltmweight = 0.1
        )
        @test res ≈ 0.1

        ## Buffer rate
        res = f(
            fill(1, 10) |> Sequence,
            Sequence([1]),
            10,
            10.0,
            collect(1.0:0.5:5.5);
            noise = 0.0,
            bufferlengthtime = 7.0,
            bufferlengthitems = 1000,
            bufferweight = 0.5,
            stmduration = 0.0,
            ltmhalflife = 1.0,
            ltmweight = 1.0
        )
        ans = (
            decayexp(2.0, 1.0, 1.0, 0.0) +
            decayexp(1.5, 1.0, 1.0, 0.0) +
            decayexp(1.0, 1.0, 1.0, 0.0) +
            decayexp(0.5, 1.0, 1.0, 0.0) +
            1.0 + 5.0 * 0.5
        )
        @test res ≈ ans
    end
end
