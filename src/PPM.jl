module PPM

export Sequence, PPMSimple, PPMDecay
export getalphabet, assequence, newppmsimple, newppmdecay, modelseq!, getweight, getngramweights

using DataFrames
using Distributions: Distribution, Univariate, Continuous, Normal
using Random: AbstractRNG, MersenneTwister, rand


include("utils.jl")
include("ppmgeneric.jl")
include("ppmsimple.jl")
include("ppmdecay.jl")
include("newmodel.jl")
include("getngramweights.jl")


end
