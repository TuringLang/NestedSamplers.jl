# NestedSamplers

[![Build Status](https://github.com/mileslucas/NestedSamplers.jl/actions)](https://github.com/mileslucas/NestedSamplers.jl/workflows/CI/badge.svg)
[![Coverage](https://codecov.io/gh//.jl/branch/master/graph/badge.svg)](https://codecov.io/gh//.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mileslucas.github.io/NestedSamplers.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mileslucas.github.io/NestedSamplers.jl/dev)

**WIP: Do Not Use**

## Control Flow

* Get samples from unit cube
* Transform from unit cube to prior space
* evaluate log likelihood at each point
* Find lowest likelihood point
* update evidence and information
* add worst object to samples
* Calculate bounding ellipsoid in prior space
* Choose point within ellipsoid until it has likelihood greater than previous lowest likelihood
* After Stopping criterion met sample without shrinking jf