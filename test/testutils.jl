@testset "utilities" begin
    @testset "last n" begin
        input = Sequence()
        push!(input, 1)
        push!(input, 2)
        push!(input, 3)

        output = PPM.lastn(input, 0)
        @test length(output) == 0

        output = PPM.lastn(input, 1)
        @test length(output) == 1
        @test output[1] == 3

        output = PPM.lastn(input, 2)
        @test length(output) == 2
        @test output[1] == 2
        @test output[2] == 3

        input = Sequence()
        output = PPM.lastn(input, 0)
        @test length(output) == 0
    end

    @testset "compute entropy" begin
        x1 = Float64[]
        push!(x1, 0.5)
        push!(x1, 0.25)
        push!(x1, 0.25)
        @test round(PPM.computeentropy(x1); digits = 3) ≈ 1.5

        x1 = Float64[]
        push!(x1, 0.1)
        push!(x1, 0.2)
        push!(x1, 0.7)
        @test round(PPM.computeentropy(x1); digits = 3) ≈ 1.157

        x1 = Float64[]
        push!(x1, 0.01)
        push!(x1, 0.09)
        push!(x1, 0.9)
        @test round(PPM.computeentropy(x1); digits = 3) ≈ 0.516
    end

    @testset "normalise distribution" begin
        input = Float64[]
        push!(input, 2.0)
        push!(input, 1.0)
        push!(input, 1.0)

        output = PPM.normalisedistribution(input)

        @test output[1] ≈ 0.5
        @test output[2] ≈ 0.25
        @test output[3] ≈ 0.25

        input = Float64[]
        output = PPM.normalisedistribution(input)
        @test length(output) == 0
    end
end
