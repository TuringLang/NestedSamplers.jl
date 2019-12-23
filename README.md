# NestedSamplers

[![Build Status](https://github.com/mileslucas/NestedSamplers.jl/workflows/CI/badge.svg)](https://github.com/mileslucas/NestedSamplers.jl/actions)
[![Coverage](https://codecov.io/gh/mileslucas/NestedSamplers.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mileslucas/NestedSamplers.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mileslucas.com/NestedSamplers.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mileslucas.com/NestedSamplers.jl/dev)

**WIP: Do Not Use**

**Primary Author:** Miles Lucas ([mileslucas](https://github.com/mileslucas))

This package was heavily influenced by [`nestle`](https://github.com/kbarbary/nestle) and [`NestedSampling.jl`](https://github.com/kbarbary/NestedSampling.jl).

## TODO

- [x] Single Ellipsoidal sampler
- [x] Multi Ellipsoidal sampler
- [x] AbstractMCMC interface
- [ ] Turing Interface (probably within Turing)
- [x] Tests
- [ ] Documentation (if appropriate, maybe just some README stuff otherwise put it  under the Turing docs)
- [ ] Optimization

## Control Flow


* Get samples from unit cube
* Transform from unit cube to prior space
* evaluate log likelihood at each point
* Find lowest likelihood point
* update evidence and information
* add worst object to samples
* For the **Single** method
  * Calculate bounding ellipsoid in prior space enlarged by some factor
  * Choose point within ellipsoid until it has likelihood greater than previous lowest likelihood
* For the **Multi** method
  * Calculate largest bounding ellipsoid in prior space
  * Find the endpoints of the major axis
  * do K-means clustering with K=2 centered on the endpoints of the major axis
  * Fit ellipsoids to each cluster
  * If the volume of both ellipsoids is less than half the parent's volume, recurse into each
  * Else, return the current parent ellipsoid
  * Then, sample within the group of ellipsoids until finding a point with greater likelihood than the previous lowest
* The final `N-nactive` points just add the current active points to sample list (no longer fitting ellipsoids)
