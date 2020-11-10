@testset "insert" begin
    @testset "n-grams should only be inserted if they fit within the temporal buffer" begin
        mod = newppmdecay(
            20;
            noise = 0.1,
            bufferweight = 1.0,
            stmweight = 1.0,
            ltmweight = 1.0,
            bufferlengthtime = 5.0,
            bufferlengthitems = 21,
            orderbound = 20,
            onlylearnfrombuffer = true
        )
        seq = Sequence(1:20)
        time = collect(0.0:19.0)
        modelseq!(mod, seq; time = time, predict = false)

        @test max(map(length, PPM.aslist(mod).ngram)...) ≈ 5


        mod = newppmdecay(
            20;
            noise = 0.1,
            bufferweight = 1.0,
            stmweight = 1.0,
            ltmweight = 1.0,
            bufferlengthtime = 5.0,
            bufferlengthitems = 21,
            orderbound = 20,
            onlylearnfrombuffer = true
        )
        seq = Sequence(1:20)
        time = collect(0.0:0.5:9.5)
        modelseq!(mod, seq; time = time, predict = false)

        @test max(map(length, PPM.aslist(mod).ngram)...) ≈ 10
    end
end
∘
