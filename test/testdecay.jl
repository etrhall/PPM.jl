@testset "decay" begin
    function testdecay(
        seq::Union{Vector, AbstractString},
        time::Vector{Float64};
        orderbound::Integer = 3,
        onlylearnfrombuffer::Bool = false,
        args...
    )
        alphabet = getalphabet(seq)
        m = newppmdecay(
            length(alphabet);
            orderbound = orderbound,
            onlylearnfrombuffer = onlylearnfrombuffer,
            args...
        )
        s = assequence(seq, alphabet)
        return modelseq!(m, s; time = time)
    end

    @testset "simple tests" begin
        seq = "abracadabra"
        ic1 = testdecay(
            seq,
            collect(1.0:11.0);
            bufferlengthitems = 20,  # <----
            bufferlengthtime = 100.0,
            noise = 0.0
        )["informationcontent"]
        ic2 = testdecay(
            seq,
            collect(1.0:11.0);
            bufferlengthitems = 10,  # <----
            bufferlengthtime = 100.0,
            noise = 0.0
        )["informationcontent"]
        @test ic1 ≈ ic2

        ic2 = testdecay(
            seq,
            collect(1.0:11.0);
            bufferlengthitems = 9,  # <----
            bufferlengthtime = 100.0,
            noise = 0.0
        )["informationcontent"]
        @test ic1 ≠ ic2

        seq = "abcabcabdabc"
        finaldist1 = testdecay(
            seq,
            collect(1.0:12.0);
            bufferlengthitems = 20,  # <---- all the sequence fits in the buffer
            bufferlengthtime = 100.0,
            noise = 0.0
        )["distribution"][end]
        finaldist2 = testdecay(
            seq,
            collect(1.0:12.0);
            bufferlengthitems = 4,  # <---- buffer of length 4
            bufferlengthtime = 100.0,
            noise = 0.0
        )["distribution"][end]
        @test finaldist1[3] > finaldist2[3]  # <--- forgetting about 'c'

        x = testdecay(
            seq,
            collect(1.0:12.0);
            bufferlengthitems = 4,  # <---- buffer of length 4
            bufferlengthtime = 100.0,
            bufferweight = 1.0,
            stmduration = 0.0,
            ltmhalflife = 1e-30,
            noise = 0.0
        )["distribution"]
        y = readdlm("./data/decay-regression-1.csv", ',')
        @test hcat(x...)' ≈ y
    end
end
