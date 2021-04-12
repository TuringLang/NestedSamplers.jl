# Correlated Gaussian

This example will explore a highly-correlated Gaussian using [`Models.CorrelatedGaussian`](@ref).

## Setup

For this example, you'll need to add the following packages
```julia
julia>]add Distributions MCMCChains Measurements NestedSamplers StatsBase StatsPlots
```

```@setup correlated
using AbstractMCMC
using Random
AbstractMCMC.setprogress!(false)
Random.seed!(8452)
```

## Define model

```@example correlated
using NestedSamplers

# set up a 5-dimensional Gaussian
D = 5
model, logz = Models.CorrelatedGaussian(D)
nothing; # hide
```

let's take a look at a couple of parameters to see what the likelihood surface looks like

```@example correlated
using StatsPlots

θ1 = range(-3, 3, length=100)
θ2 = range(-3, 3, length=100)
f = [model.loglike([t1, t2, 0, 0, 0]) for t2 in θ2, t1 in θ1]
contourf(θ1, θ2, f,
    aspect_ratio=1,
    xlims=(-3, 3),
    ylims=(-3, 3),
    xlabel="θ1",
    ylabel="θ2")
```

## Sample

```@example correlated
using MCMCChains
using StatsBase
sampler = Nested(D, 50D; 
    bounds=Bounds.Ellipsoid,
    proposal=Proposals.Slice()
)
names = ["θ_$i" for i in 1:D]
chain, state = sample(model, sampler; dlogz=0.01, param_names=names)
# resample chain using statistical weights
chain_resampled = sample(chain, Weights(vec(chain[:weights])), length(chain));
nothing # hide
```

## Results

```@example correlated
chain_resampled
```

```@example correlated
corner(chain_resampled)
```

```@example correlated
using Measurements
logz_est = state.logz ± state.logzerr
diff = logz_est - logz
print("logz: ", logz, "\nestimate: ", logz_est, "\ndiff: ", diff)
```
