@testset "noise" begin
    @testset "misc" begin
        mod = newppmdecay(20; noise = 0.1, ltmhalflife = 1e60)
        seq = Sequence([1, 2, 3])
        time = collect(0.0:2.0)
        modelseq!(mod, seq; time = time, predict = false)

        y = [getweight(mod, Sequence([1, 2]), 2, 2.0, true) for i in 1:1e5]

        @test isapprox(
            mean(y),
            1 + mod.noisemean,
            atol = 1e-2
        )
    end
end
