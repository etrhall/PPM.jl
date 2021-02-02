module PPM

export Sequence, PPMSimple, PPMDecay, SequencePrediction
export getalphabet, assequence, newppmsimple, newppmdecay, modelseq!, getweight, getngramweights

using DataFrames
using Distributions: Distribution, Univariate, Continuous, Normal
using PrettyTables
using Random: AbstractRNG, MersenneTwister, rand


include("utils.jl")
include("ppmgeneric.jl")
include("ppmsimple.jl")
include("ppmdecay.jl")
include("ppmpoly.jl")
include("newmodel.jl")
include("getngramweights.jl")


end
