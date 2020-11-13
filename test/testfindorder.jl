@testset "find order" begin
    @testset "buffer models" begin
        # With an buffer length of 2 seconds,
        # and with events every 1 second,
        # the maximum model order should be 2.
        model = newppmdecay(
            10;
            bufferlengthtime = 2.0,
            bufferlengthitems = 15,
            onlylearnfrombuffer = true,
            onlypredictfrombuffer = true
        )
        result = modelseq!(
            model,
            fill(1, 10) |> Sequence;
            time = collect(1.0:10.0)
        )["modelorder"] |> maximum
        @test result == 2

        # Likewise for a 3-second buffer:
        model = newppmdecay(
            10;
            bufferlengthtime = 3.0,
            bufferlengthitems = 15,
            onlylearnfrombuffer = true,
            onlypredictfrombuffer = true
        )
        result = modelseq!(
            model,
            fill(1, 10) |> Sequence;
            time = collect(1.0:10.0)
        )["modelorder"] |> maximum
        @test result == 3

        # A 1.9-second buffer should yield an order bound of 1.
        model = newppmdecay(
            10;
            bufferlengthtime = 1.9,
            bufferlengthitems = 15,
            onlylearnfrombuffer = true,
            onlypredictfrombuffer = true
        )
        result = modelseq!(
            model,
            fill(1, 10) |> Sequence;
            time = collect(1.0:10.0)
        )["modelorder"] |> maximum
        @test result == 1

        # Now multiplying everything by 10:
        model = newppmdecay(
            10;
            bufferlengthtime = 20.0,
            bufferlengthitems = 15,
            onlylearnfrombuffer = true,
            onlypredictfrombuffer = true
        )
        result = modelseq!(
            model,
            fill(1, 10) |> Sequence;
            time = collect(10.0:10.0:100.0)
        )["modelorder"] |> maximum
        @test result == 2

        # If we disable 'only_predict_from_buffer', these things shouldn't matter.
        model = newppmdecay(
            10;
            bufferlengthtime = 2.0,
            bufferlengthitems = 2,
            onlypredictfrombuffer = false
        )
        result = modelseq!(
            model,
            fill(1, 20) |> Sequence;
            time = collect(1.0:20.0)
        )["modelorder"] |> maximum
        @test result == 10
    end
end
