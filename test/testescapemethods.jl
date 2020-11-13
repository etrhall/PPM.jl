@testset "escpe methods" begin
    function test(escape::AbstractString, updateexclusion::Bool)
        mod = newppmsimple(
            5;
            escape = escape,
            updateexclusion = updateexclusion,
            shortestdeterministic = true,
            orderbound = 100
        )
        res = modelseq!(
            mod,
            assequence("abracadabra", (a = 1, b = 2, c = 3 , d = 4, r = 5))
        )["distribution"]
        name = updateexclusion ? "escape-$escape-update-excluded" : "escape-$escape"
        ans = readdlm("./data/$name.csv", ','; skipstart = 1)
        @test isapprox(hcat(res...)', ans, atol = 1e-2)
    end

    # These regression tests come from IDyOM v 1.5 (Pearce, 2005)
    @testset "different escape methods, without update exclusion" begin
        test("a", false)
        test("b", false)
        test("c", false)
        test("d", false)
        test("ax", false)
    end

    # Note! The LISP implementation of Pearce (2005) has mistakes
    # in the implementation of update exclusion.
    # These regression tests come from the latest version of mtp_development
    # (as of Jan 2020) which has fixed these problems.
    @testset "different escape methods, without update exclusion" begin
        test("a", true)
        test("b", true)
        test("c", true)
        test("d", true)
        test("ax", true)
    end
 end
