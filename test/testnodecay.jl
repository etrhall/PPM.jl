@testset "no decay" begin
    function testppm(seqs, answer; tolerance = 1e-4, alphabet = Int[], args...)
        if isempty(alphabet)
            alphabet = getalphabet([(seqs...)...])
        end
        model = newppmsimple(length(alphabet); args...)
        res = Vector{Vector{Vector{Float64}}}(undef, length(seqs))
        for i in 1:length(seqs)
            seq = assequence(seqs[i], alphabet)
            res[i] = modelseq!(model, seq)["distribution"]
        end
        for i in 1:length(res)
            @test length(res[i]) == length(answer[i])
            for j in 1:length(res[i])
                x = res[i][j]
                y = answer[i][j]
                @test isapprox(x, y; atol = tolerance)
            end
        end
    end

    function testppm2(seqs, answer; tolerance = 1e-4, alphabet = Int[], args...)
        if isempty(alphabet)
            alphabet = getalphabet([(seqs...)...])
        end
        model = newppmsimple(length(alphabet); args...)
        res = fill(NaN, length(seqs))
        for i in 1:length(seqs)
            seq = assequence(seqs[i], alphabet)
            tmp = modelseq!(model, seq)
            res[i] = mean(tmp["informationcontent"])
        end
        @test isapprox(res, answer; atol = tolerance)
    end

    @testset "PPM* dataset 1" begin
        abra = ["a", "b", "r", "a", "c", "a", "d", "a", "b", "r", "a"]
        answer = [
            [
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.6, 0.1, 0.1, 0.1, 0.1],
                [0.3333, 0.3333, 0.1111, 0.1111, 0.1111],
                [0.25, 0.25, 0.125, 0.125, 0.25],
                [0.2000, 0.5333, 0.0667, 0.0667, 0.1333],
                [0.2667, 0.2, 0.2, 0.1333, 0.2],
                [0.2083, 0.2917, 0.2917, 0.0833, 0.1250],
                [0.2500, 0.1875, 0.1875, 0.1875, 0.1875],
                [0.2093, 0.2171, 0.2171, 0.2171, 0.1395],
                [0.1915, 0.1489, 0.1277, 0.1277, 0.4043],
                [0.4348, 0.1522, 0.1304, 0.1304, 0.1522]
            ],
            [
                [0.2777778, 0.19444445, 0.16666667, 0.16666667, 0.19444445],
                [0.20245397, 0.28834355, 0.19018403, 0.19018403, 0.12883434],
                [0.15714285, 0.11428571, 0.08571428, 0.08571428, 0.55714285],
                [0.5692308, 0.12307692, 0.09230768, 0.09230768, 0.12307692],
                [0.11999999, 0.23, 0.46, 0.10999999, 0.07999999],
                [0.47272727, 0.14545456, 0.12727273, 0.10909091, 0.14545456],
                [0.111428574, 0.20571429, 0.15142857, 0.46285713, 0.068571426],
                [0.48275864, 0.13793103, 0.12068966, 0.12068966, 0.13793103],
                [0.11351351, 0.5243243, 0.14864865, 0.14864865, 0.06486486],
                [0.12612611, 0.08108108, 0.06306306, 0.06306306, 0.6666666],
                [0.6701031, 0.09278351, 0.072164945, 0.072164945, 0.09278351]
            ]
        ]
        testppm([abra, abra], answer; tolerance = 1e-3, updateexclusion = false)
    end

    @testset "IDyOM PPM* dataset 1 + 2" begin
        abra = ["a", "b", "r", "a", "c", "a", "d", "a", "b", "r", "a"]
        answer = [
            [
                [0.19999999, 0.19999999, 0.19999999, 0.19999999, 0.19999999],
                [0.59999996, 0.09999999, 0.09999999, 0.09999999, 0.09999999],
                [0.33333334, 0.33333334, 0.11111111, 0.11111111, 0.11111111],
                [0.25, 0.25, 0.125, 0.125, 0.25],
                [0.20000002, 0.53333336, 0.06666668, 0.06666668, 0.13333336],
                [0.26666665, 0.19999997, 0.19999997, 0.13333331, 0.19999997],
                [0.20833334, 0.2916667, 0.2916667, 0.083333336, 0.125],
                [0.25, 0.1875, 0.1875, 0.1875, 0.1875],
                [0.2093023, 0.21705426, 0.21705426, 0.21705426, 0.13953489],
                [0.19148937, 0.14893615, 0.12765956, 0.12765956, 0.40425533],
                [0.4347826, 0.1521739, 0.13043477, 0.13043477, 0.1521739]
            ],
            [
                [0.2777778, 0.19444445, 0.16666667, 0.16666667, 0.19444445],
                [0.20245397, 0.28834355, 0.19018403, 0.19018403, 0.12883434],
                [0.15714285, 0.11428571, 0.08571428, 0.08571428, 0.55714285],
                [0.5692308, 0.12307692, 0.09230768, 0.09230768, 0.12307692],
                [0.11999999, 0.23, 0.46, 0.10999999, 0.07999999],
                [0.47272727, 0.14545456, 0.12727273, 0.10909091, 0.14545456],
                [0.111428574, 0.20571429, 0.15142857, 0.46285713, 0.068571426],
                [0.48275864, 0.13793103, 0.12068966, 0.12068966, 0.13793103],
                [0.11351351, 0.5243243, 0.14864865, 0.14864865, 0.06486486],
                [0.12612611, 0.08108108, 0.06306306, 0.06306306, 0.6666666],
                [0.6701031, 0.09278351, 0.072164945, 0.072164945, 0.09278351]
            ],
            [
                [0.31914893, 0.19148934, 0.14893617, 0.14893617, 0.19148934],
                [0.15789472, 0.35197368, 0.20065789, 0.20065789, 0.088815786],
                [0.107382536, 0.06711409, 0.04697986, 0.04697986, 0.7315436],
                [0.73015875, 0.07936508, 0.055555556, 0.055555556, 0.07936508],
                [0.06447536, 0.17699116, 0.6384324, 0.082174465, 0.037926678],
                [0.61956525, 0.10869565, 0.08695652, 0.07608695, 0.10869565],
                [0.06033518, 0.16201115, 0.103910595, 0.64022344, 0.033519547],
                [0.625, 0.10416666, 0.08333333, 0.08333333, 0.10416666],
                [0.06620208, 0.67595816, 0.11149825, 0.11149825, 0.034843203],
                [0.090047404, 0.052132707, 0.037914697, 0.037914697, 0.78199047],
                [0.7790698, 0.0639535, 0.046511635, 0.046511635, 0.0639535]
            ]
        ]
        testppm([abra, abra, abra], answer; updateexclusion = false)
    end

    @testset "IDyOM PPM* dataset 3" begin
        seqs = [
            "abracadabra",
            "letlettertele",
            "assanissimassa",
            "mississippi",
            "wooloobooloo"
        ]
        file = readdlm("./data/idyom-3.csv", ','; skipstart = 1)
        answer = [
            [i[2:end] for i in eachrow(file) if i[1] == j]
            for j in unique(file[:, 1])
        ]
        testppm(seqs, answer; updateexclusion = false)
    end

    @testset "IDyOM update exclusion" begin
        @testset "'abracadabra' x2" begin
            seqs = ["abracadabra", "abracadabra"]
            answer = [2.4509745, 1.1457417]
            testppm2(seqs, answer; updateexclusion = true)
        end

        @testset "'abracadabra' x3" begin
            seqs = ["abracadabra", "abracadabra", "abracadabra"]
            answer = [2.4509745, 1.1457417, 0.8534344]
            testppm2(seqs, answer; updateexclusion = true)
        end

        @testset "'abracadabra' 'bracadabra'" begin
            seqs = ["abracadabra", "bracadabra"]
            answer = [2.4509745, 1.1973304]
            testppm2(seqs, answer; updateexclusion = true)
        end

        @testset "'abracadabra' 'abrabrac'" begin
            seqs = ["abracadabra", "abrabrac"]
            answer = [2.4509745, 1.4018788]
            testppm2(seqs, answer; updateexclusion = true)
        end

        @testset "'abracadabra' 'abratbrac'" begin
            seqs = ["abracadabra", "abratbrac"]
            answer = [2.591134, 1.8791169]
            testppm2(seqs, answer; updateexclusion = true)
        end

        @testset "'abracadabra' 'abrabtrac'" begin
            seqs = ["abracadabra", "abrabtrac"]
            answer = [2.591134, 2.0285676]
            testppm2(seqs, answer; updateexclusion = true)
        end

        @testset "'abracadabra' 'abrabrtac'" begin
            seqs = ["abracadabra", "abrabrtac"]
            answer = [2.591134, 1.8929263]
            testppm2(seqs, answer; updateexclusion = true)
        end

        @testset "'abracadabra' 'abrabratc'" begin
            seqs = ["abracadabra", "abrabratc"]
            answer = [2.591134, 1.7575804]
            testppm2(seqs, answer; updateexclusion = true)
        end

        @testset "'abracadabra' 'abrabract'" begin
            seqs = ["abracadabra", "abrabract"]
            answer = [2.591134, 1.5876032]
            testppm2(seqs, answer; updateexclusion = true)
        end
    end

    @testset "IDyOM PPM* with mixtures" begin
        seqs = [
            "abracadabrac",
            "letlettertele",
            "assanissimassa",
            "mississippi",
            "agcgacgag"
        ]
        for i in 1:length(seqs)
            file = readdlm("./data/ppm-mix-$i.csv", ','; skipstart = 1)
            answer = [[j for j in eachrow(file)]]
            testppm([seqs[i]], answer; updateexclusion = false)
        end
    end

    @testset "IDyOM PPM* with mixtures and update exclusion" begin
        seqs = ["abracadabrac"]
        for i in 1:length(seqs)
            file = readdlm("./data/ppm-mix-ui-$i.csv", ','; skipstart = 1)
            answer = [[j for j in eachrow(file)]]
            testppm([seqs[i]], answer; updateexclusion = true)
        end
    end

    @testset "IDyOM PPM with mixtures and order bound = 0, 1" begin
        seqs = ["abracadabrac", "abracadabrac"]
        for i in 1:length(seqs)
            file = readdlm("./data/ppm-fix-$(i-1).csv", ','; skipstart = 1)
            answer = [[j for j in eachrow(file)]]
            testppm([seqs[i]], answer; orderbound = i - 1, updateexclusion = false)
        end
    end

    @testset "Louis Couperin: unmeasured prelude, no. 7" begin
        seqs = [[
            45, 52, 57, 60, 64, 69, 45, 52, 57, 60, 64, 69, 45, 45, 44, 52, 60,
            60, 59, 64, 64, 71, 45, 57, 76, 74, 72, 71, 72, 72, 72, 72, 72, 72,
            72, 72, 52, 64, 69, 69, 71, 71, 71, 71, 71, 71, 50, 52, 54, 55, 52,
            53, 52, 50, 48, 48, 47, 69, 71, 72, 55, 74, 62, 67, 48, 67, 69, 71,
            57, 72, 72, 57, 64, 65, 72, 72, 72, 60, 62, 66, 71, 71, 62, 68, 66,
            68, 64, 64, 64, 64, 59, 59, 60, 64, 68, 68, 69, 69, 50, 49, 49, 50,
            59, 64, 64, 65, 69, 47, 48, 50, 52, 53, 50, 50, 52, 60, 64, 69, 52,
            60, 60, 59, 69, 69, 68, 68, 68, 45, 52, 57, 61, 64, 69, 45, 52, 57,
            62, 61, 64, 69
        ]]
        answer = [4.195832]
        testppm2(seqs, answer, updateexclusion = false)
    end

    @testset "Paul Hindemith: Acht Stücke Für Flöte Allein: VI. Lied, Leicht Bewegt" begin
        seqs = [[
            84, 83, 85, 86, 81, 82, 81, 80, 80, 80, 79, 77, 79, 76, 83, 73, 72,
            71, 72, 77, 76, 75, 77, 79, 80, 82, 84, 85, 87, 88, 88, 87, 82, 77,
            78, 83, 81, 79, 77, 76, 75, 77, 78, 73, 74, 72, 71, 71, 70, 69, 68,
            67, 68, 70, 62, 73, 74, 63, 78, 79, 92, 90, 89, 88, 88, 91, 87, 86,
            85, 84, 83, 85, 86, 81, 82, 81, 80, 80, 80, 79, 77, 79, 76, 83, 73,
            71, 73, 74, 74, 74, 73, 71, 73, 70, 77, 67, 68, 69, 70, 70, 68, 69,
            70, 70, 68, 69, 70, 63, 63, 68, 66, 63, 62, 70, 63, 68, 66, 63, 62,
            70, 63, 68, 66, 63, 62
        ]]
        answer = [4.210231]
        testppm2(seqs, answer, updateexclusion = false)
    end
end
