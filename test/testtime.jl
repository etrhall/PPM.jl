@testset "time" begin
    @testset "times must be in increasing order" begin
        err = "decreasing values of time are not permitted"
        mod = newppmdecay(10)
        seq = Sequence(rand(1:5, 10))
        time = collect(10.0:-1.0:1.0)
        @test_throws ErrorException(err) modelseq!(mod, seq; time = time)

        err = "a sequence may not begin before the previous sequence finished"
        mod = newppmdecay(10)
        seq = Sequence(rand(1:5, 10))
        time1 = collect(1.0:10.0)
        time2 = collect(9.0:18.0)
        modelseq!(mod, seq; time = time1)
        @test_throws ErrorException(err) modelseq!(mod, seq; time = time2)
    end
end
