using NestedSamplers
using AbstractMCMC
using MCMCChains
using StatsBase
using Distributions
using StatsPlots

#----------------------------
# Definitions

@info "Defining likelihood"
const tmax = 5π
const constant = -2log(tmax)

function loglike(x)
    t = @. 2tmax * x - tmax
    return (2cos(t[1]/2) * cos(t[2]/2))^5
end

loglike(x, y) = loglike([x, y])

priors = [Uniform(0, 1), Uniform(0, 1)]

#----------------------------
# Plotting

@info "Initial plots"
default(c=:blues_r, aspect_ratio=:equal, xlims=(0,1), ylims=(0,1), markerstrokealpha=0.0)
x = range(0, 1, length=1000)
p = heatmap(x, x, loglike, title="True log-likelihood surface")
savefig(joinpath(@__DIR__, "eggshell.png"))

#----------------------------
# Sampling

@info "Sampling"
model = NestedModel(loglike, priors)
spl = Nested(200, update_interval=20, method=:multi)
@time chain = sample(model, spl, 2000; param_names=["x", "y"])

#----------------------------
# Posterior plots
@info "Posterior Plots"
xs = Array(chain[:x])
ys = Array(chain[:y])
p = scatter(xs, ys, alpha=0.4, title="Posterior Samples", label="")
savefig(joinpath(@__DIR__, "eggshell-posterior.png"))


