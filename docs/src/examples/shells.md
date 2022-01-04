# Gaussian Shells

This example will explore the classic Gaussian shells model using [`Models.GaussianShells`](@ref).

## Setup

For this example, you'll need to add the following packages
```julia
julia>]add Distributions MCMCChains Measurements NestedSamplers StatsBase StatsPlots
```

```@setup shells
using AbstractMCMC
using Random
AbstractMCMC.setprogress!(false)
Random.seed!(8452)
```

## Define model

```@example shells
using NestedSamplers

model, logz = Models.GaussianShells()
nothing; # hide
```

let's take a look at a couple of parameters to see what the likelihood surface looks like

```@example shells
using StatsPlots

x = range(-6, 6, length=1000)
y = range(-2.5, 2.5, length=1000)
loglike = model.prior_transform_and_loglikelihood.loglikelihood
logf = [loglike([xi, yi]) for yi in y, xi in x]
heatmap(
    x, y, exp.(logf),
    xlims=extrema(x),
    ylims=extrema(y),
    xlabel="x",
    ylabel="y",
)
```

## Sample

```@example shells
using MCMCChains
using StatsBase
# using multi-ellipsoid for bounds
# using default rejection sampler for proposals
sampler = Nested(2, 1000)
chain, state = sample(model, sampler; dlogz=0.05, param_names=["x", "y"])
# resample chain using statistical weights
chain_resampled = sample(chain, Weights(vec(chain[:weights])), length(chain));
nothing # hide
```

## Results

```@example shells
chain_resampled
```

```@example shells
marginalkde(chain[:x], chain[:y])
plot!(xlims=(-6, 6), ylims=(-2.5, 2.5), sp=2)
plot!(xlims=(-6, 6), sp=1)
plot!(ylims=(-2.5, 2.5), sp=3)
```

```@example shells
density(chain_resampled)
vline!([-5.5, -1.5, 1.5, 5.5], c=:black, ls=:dash, sp=1)
vline!([-2, 2], c=:black, ls=:dash, sp=2)
```

```@example shells
using Measurements
logz_est = state.logz ± state.logzerr
diff = logz_est - logz
println("logz: $logz")
println("estimate: $logz_est")
println("diff: $diff")
nothing # hide
```
