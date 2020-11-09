"""
getngramweights(
    ppm::PPMSimple,
    order::Integer,
    pos::Integer,
    time::Real
)


Tabulates weights for all possible n-grams of a given order.

`ppm`:
A PPM model object as produced by (for example)
``newppmsimple`` or ``newppmdecay``,
and subsequently trained on input sequences using modelseq}}.

`order`:
The order (i.e. number of symbols) of the n-grams to retrieve.

`pos`:
The nominal 'position' at which the n-gram counts are retrieved
(only relevant for decay-based models).

`time`:
The nominal 'time' at which the n-grams are retrieved
(only relevant for decay-based models).

`zeroindexed`:
Whether the n-grams should be presented as zero-indexed (as opposed to one-indexed)
integer vectors.

Returns a data frame where each row corresponds to an n-gram.
These n-grams are exhaustively enumerated from all possible symbols drawn from the model's alphabet.
The tibble contains n columns 'elt1', 'elt1', ... 'eltn',
corresponding to the n symbols in the n-gram,
and a column 'weight', identifying the weight of the specified n-gram.
"""
function getngramweights(
    ppm::PPMSimple,
    order::Integer,
    pos::Integer,
    time::Real
)
    prods = collect(Base.product((1:ppm.alphabetsize for i in 1:order)...))[:]
    res = DataFrame(prods)
    rename!(res, [Symbol("elt$i") for i in 1:order])

    insertcols!(
        res,
        order + 1,
        :weight => map(
            x->PPM.getweight(ppm, Sequence([x...]), pos, time, false),
            prods
        )
    )

    return res
end
