@testset "escape methods 2" begin
    function test(escape::AbstractString, updateexclusion::Bool)
        alphabet = getalphabet([
            'a', 'b', 'c', 'd', 'e', 'i', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'w'
        ])
        seqs = [
            "abracadabra",
            "letlettertele",
            "assanissimassa",
            "mississippi",
            "wooloobooloo"
        ] |> x->map(y->assequence(y, alphabet), x)

        mod = newppmsimple(
            length(alphabet);
            escape = escape,
            updateexclusion = updateexclusion,
            shortestdeterministic = true,
            orderbound = 100
        )

        res = map(s->modelseq!(mod, s)["distribution"], seqs)

        name = updateexclusion ? "escape-v2-$escape-update-excluded" : "escape-v2-$escape"
        file = readdlm("./data/$name.csv", ','; skipstart = 1)
        ans = [
            [i[2:end] for i in eachrow(file) if i[1] == j]
            for j in unique(file[:, 1])
        ]

        @test isapprox(res, ans, atol = 1e-5)
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
    @testset "different escape methods, with update exclusion" begin
        test("a", true)
        test("b", true)
        test("c", true)
        test("d", true)
        test("ax", true)
    end
end
