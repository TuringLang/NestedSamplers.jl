# Benchmarks

The following benchmarks show the performance of NestedSamplers.jl. As with any statistical inference package, the likelihood function will often dominate the runtime. This is important to consider when comparing packages across different languages- in general a custom Julia likelihood function may be faster than the same code written in Python/numpy. As an example, compare the relative timings of these two simple Guassian likelihoods

```julia
using BenchmarkTools
using PyCall

# julia version
gauss_loglike(X) = sum(x -> exp(-0.5 * x^2) / sqrt(2π), X)

# python version
py"""
import numpy as np
def gauss_loglike(X):
    return np.sum(np.exp(-0.5 * X ** 2) / np.sqrt(2 * np.pi))
"""
gauss_loglike_py = py"gauss_loglike"
xs = randn(100)
```

```julia
@btime gauss_loglike($xs)
```

```
  611.971 ns (0 allocations: 0 bytes)
26.813747896467206
```

```julia

@btime gauss_loglike_py($xs)
```

```
  13.129 μs (6 allocations: 240 bytes)
26.81374789646721
```

In certain cases, you can use language interop tools (like [PyCall.jl](https://github.com/JuliaPy/PyCall.jl)) to use Julia likelihoods with Python libraries.

## Setup and system information

The benchmark code can be found in the [`bench`](https://github.com/TuringLang/NestedSamplers.jl/blob/main/bench/) folder. The system information at the time these benchmarks were ran is

```julia
julia> versioninfo()
Julia Version 1.7.1
Commit ac5cc99908* (2021-12-22 19:35 UTC)
Platform Info:
  OS: macOS (x86_64-apple-darwin20.5.0)
  CPU: Intel(R) Core(TM) i5-8259U CPU @ 2.30GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, skylake)
Environment:
  JULIA_NUM_THREADS = 1
```

## Highly-correlated multivariate Guassian

This benchmark uses [`Models.CorrelatedGaussian`](@ref) and simply measures the time it takes to fully sample down to `dlogz=0.01`. This benchmark is exactly the same as the benchmark detailed in the [JAXNS paper](https://ui.adsabs.harvard.edu/abs/2020arXiv201215286A/abstract).

### Timing

```@example sample-benchmark
using CSV, DataFrames, Plots # hide
benchdir = joinpath(dirname(pathof(NestedSamplers)), "..", "bench") # hide
results = DataFrame(CSV.File(joinpath(benchdir, "sampling_results.csv"))) # hide
plot(results.D, results.t, label="NestedSamplers.jl", marker=:o, yscale=:log10 # hide
    ylabel="runtime (s)", xlabel="prior dimension", leg=:topleft) # hide
```

### Accuracy

The following shows the Bayesian evidence estmiate as compared to the true value

```@example sample-benchmark
plot(results.D, results.dlnZ, yerr=results.lnZstd, label="NestedSamplers.jl", # hide
    marker=:o, ylabel="ΔlnZ", xlabel="prior dimension", leg=:topleft) # hide
hlines([0.0], c=:black, ls=:dash, alpha=0.7, label="") # hide
```
