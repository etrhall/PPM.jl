"""
newppmsimple(
    alphabetsize::Integer,
    orderbound::Integer = 10,
    shortestdeterministic::Bool = true,
    exclusion::Bool = true,
    updateexclusion::Bool = true,
    escape::AbstractString = "c",
    debugsmooth::Bool = true
)


Creates a simple PPM model, that is, a PPM model
without any non-traditional features such as memory decay.

`alphabetsize`:
The size of the alphabet upon which the model will be trained and tested.

`orderbound`:
The model's Markov order bound. For example, an order bound of two means
that the model makes predictions based on the two preceding symbols.

`shortestdeterministic`:
If ``true``, the model will 'select' the shortest available order
that provides a deterministic prediction, if such an order exists,
otherwise defaulting to the longest available order.
For a given prediction, if this rule results in a lower model order
than would have otherwise been selected,
then full counts (not update-excluded counts) will be used for
the highest model order (but not for lower model orders).
This behaviour matches the implementations of PPM* in
Pearce (2005) Bunton (1996).

`exclusion`:
If ``true``, implements exclusion as defined in
Pearce (2005) Bunton (1996).

`updateexclusion`:
If ``true``, implements update exclusion as defined in
Pearce (2005) Bunton (1996).

`escape`:
Takes values ``"a"``, ``"b"``, ``"c"``, ``"d"`` or ``"ax"``,
corresponding to the eponymous escape methods
in Pearce (2005).
Note that there is a mistake in the definition of escape method
"AX" in Pearce (2005)
the denominator of lambda needs to have 1 added.
This is what we implement here. Note that Pearce's
LISP implementation correctly adds 1 here, like us.

`debugsmooth`:
Whether to print (currently rather messy and ad hoc) debug output
for smoothing.

> *Note*:
> The implementation does not scale well to very large order bounds (> 50).

Returns a PPM model object.
These objects have reference semantics.
"""
function newppmsimple(
    alphabetsize::Integer,
    orderbound::Integer = 10,
    shortestdeterministic::Bool = true,
    exclusion::Bool = true,
    updateexclusion::Bool = true,
    escape::AbstractString = "c",
    debugsmooth::Bool = true
)
    validescapemethods = ("a", "b", "c", "d", "ax")
    if escape ∉ validescapemethods
        error("escape parameter must be one of: ", join(validescapemethods, ", "))
    end

    return PPMSimple(
        alphabetsize,
        orderbound,
        shortestdeterministic,
        exclusion,
        updateexclusion,
        escape,
        debugsmooth
    )
end


"""
newppmdecay(
    alphabetsize::Integer,
    orderbound::Integer = 10,
    ltmweight::Real = 1,
    ltmhalflife::Real = 10,
    ltmasymptote::Real = 0,
    noise::Real = 0,
    stmweight::Real = 1,
    stmduration::Real = 0,
    bufferweight::Real = 1,
    bufferlengthtime::Real = 0,
    bufferlengthitems::Integer = 0,
    onlylearnfrombuffer::Bool = false,
    onlypredictfrombuffer::Bool = false,
    seed = rand(Int, 1),
    debugsmooth::Bool = false,
    debugdecay::Bool = false
)

Creates a decay-based PPM model.

Decay-based PPM models generalise the PPM algorithm to incorporate
memory decay, where the effective counts of observed n-grams
decrease over time to reflect processes of auditory memory.

The weight of a given n-gram over time is determined by a decay kernel.
This decay kernel is parametrised by the arguments
``w0``, ``w1``, ``w2``, ``w∞``,
``nb``, ``tb`` ``t1``, ``t2``, ``σϵ``
(see above).

The decay kernel has three phases:

- Buffer
- Short-term memory
- Long-term mermory

While within the buffer, the n-gram has weight ``w0``.
The buffer has limited temporal and itemwise capacity.
In particular, an n-gram will leave the buffer once one
of two conditions is satisfied:

- A set amount of time, ``tb``, elapses since the first symbol in the n-gram was observed, or
- The buffer exceeds the number of symbols it can store, ``nb``,
and the n-gram no longer fits completely in the buffer,
having been displaced by new symbols.

There are some subtleties about how this actually works in practice,
refer to Harrison (2020) for details.

The second phase, short-term memory, begins as soon as the
buffer phase completes. It has a fixed temporal duration
of ``t1``. At the beginning of this phase,
the n-gram has weight ``w1``;
during this phase, its weight decays exponentially until it reaches
``w2`` at timepoint ``t2``.

The second phase, long-term memory, begins as soon as the
short-term memory phase completes. It has an unlimited temporal duration.
At the beginning of this phase,
the n-gram has weight ``w2``;
during this phase, its weight decays exponentially
to an asymptote of ``w∞``.

The model optionally implements Gaussian noise at the weight retrieval stage.
This Gaussian is parametrised by the standard deviation parameter
σϵ.
See Harrison (2020) for details.

This function supports simpler decay functions with fewer stages;
in fact, the default parameters define a one-stage decay function,
corresponding to a simple exponential decay with a half life of 10 s.
To enable the buffer, `bufferlengthtime` and `bufferlengthitems`
should be made non-zero, and `onlylearnfrombuffer` and
`onlypredictfrombuffer` should be set to ``true``.
Likewise, retrieval noise is enabled by setting `noise` to a non-zero value,
and the short-term memory phase is enabled by setting `stmduration`
to a non-zero value.

The names of the 'short-term memory' and 'long-term memory' phases
should be considered arbitrary in this context;
they do not necessarily correspond directly to their
psychological namesakes, but are instead simply terms of convenience.

The resulting PPM-Decay model uses interpolated smoothing with escape method A,
and explicitly disables exclusion and update exclusion.
See Harrison (2020) for details.

`alphabetsize:`
The size of the alphabet from which sequences are drawn.

`orderbound`:
The model's Markov order bound.

`ltmweight`:
``w2``, initial weight in the long-term memory phase.

`ltmhalflife`:
``t2``, half life of the long-term memory phase.
Must be greater than zero.

`ltmasymptote`:
``w∞``, asymptotic weight as time tends to infinity.

`noise`:
``σϵ``, scale parameter for the retrieval noise distribution.

`stmweight`:
``w1``, initial weight in the short-term memory phase.

`stmduration`:
``t1``, temporal duration of the short-term memory phase, in seconds.

`bufferweight`:
``w0``, weight during the buffer phase.

`bufferlengthtime`:
``nb``, the model's temporal buffer capacity.

`bufferlengthitems`:
``tb``, the model's itemwise buffer capacity.

`onlylearnfrombuffer`:
If ``true``, then n-grams are only learned if they fit within
the memory buffer.

`onlypredictfrombuffer`:
If ``true``, then the context used for prediction is limited by the memory buffer.
Specifically, for a context to be used for prediction,
the first symbol within that context must still be within the buffer
at the point immediately before the predicted event occurs.

`seed`:
Random seed for prediction generation.
By default this is linked with Julia's random seed, such that
reproducible behaviour can be ensured as usual with the
``seed!`` function.

`debugsmooth`:
Whether to print (currently rather messy and ad hoc) debug output
for the smoothing mechanism.

`debugdecay`
Whether to print (currently rather messy and ad hoc) debug output
for the decay mechanism.

Returns a PPM-decay model object.
These objects have reference semantics.
"""
function newppmdecay(
    alphabetsize::Integer,
    orderbound::Integer = 10,
    ltmweight::Real = 1,
    ltmhalflife::Real = 10,
    ltmasymptote::Real = 0,
    noise::Real = 0,
    stmweight::Real = 1,
    stmduration::Real = 0,
    bufferweight::Real = 1,
    bufferlengthtime::Real = 0,
    bufferlengthitems::Integer = 0,
    onlylearnfrombuffer::Bool = false,
    onlypredictfrombuffer::Bool = false,
    seed = rand(Int),
    debugsmooth::Bool = false,
    debugdecay::Bool = false
)
    decaypar = (
        ltmweight = Float64(ltmweight),
        ltmhalflife = Float64(ltmhalflife),
        ltmasymptote = Float64(ltmasymptote),
        noise = Float64(noise),
        stmweight = Float64(stmweight),
        stmduration = Float64(stmduration),
        bufferweight = Float64(bufferweight),
        bufferlengthtime = Float64(bufferlengthtime),
        bufferlengthitems = bufferlengthitems,
        onlylearnfrombuffer = onlylearnfrombuffer,
        onlypredictfrombuffer = onlypredictfrombuffer,
    )

    return PPMDecay(
        alphabetsize,
        orderbound,
        decaypar,
        seed,
        debugsmooth,
        debugdecay
    )
end
