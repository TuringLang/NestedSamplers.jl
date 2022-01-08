---
title: 'NestedSamplers.jl: Composable Nested Sampling in Julia'
tags:
  - Julia
  - statistics
  - bayesian-statistics
  - mcmc
authors:
  - name: Miles Lucas
    orcid: 0000-0001-6341-310X
    affiliation: 1
  - name: Saranjeet Kaur Bhogal
    orcid: 0000-0002-7038-1457
    affiliation: 2
  - name: Hong Ge
    orcid: 0000-0001-9421-2677
    affiliation: 3
affiliations:
 - name: Institute for Astronomy, University of Hawaii, Honolulu, HI, USA
   index: 1
 - name: Department of Statistics, Savitribai Phule Pune University, Pune, India
   index: 2
 - name: University of Cambridge, Cambridge, UK
   index: 3
date: 1/8/2021
bibliography: paper.bib
---

# Summary

Nested sampling is a method for estimating the Bayesian evidence [@skillingNestedSampling2004]. The core of the algorithm is rooted in integrating iso-likelihood "shells" in the prior space. This simultaneously produces the Bayesian evidence and weighted posterior samples. In contrast, Markov Chain Monte Carlo (MCMC) only generates samples proportional to the posterior. Nested sampling also has a variety of appealing statistical properties, which include: well-defined stopping criteria, independently generated samples, flexibility to model complex, multi-modal distributions, and direct measurements of the statistical and sampling uncertainties from a single run.

The basic algorithm can be described by the quadrature estimate of dynamically evolving points in the prior space. The Bayesian evidence is an integral, and the nested sampling algorithm estimates it with a sum of discrete measurements, represented by an array of points. The nested sampling algorithm successfully removes the point with the lowest likelihood and replaces it with a point of equal or higher likelihood. At each iteration, the full set of "live" points describes a volume, which has shrunk by the removal of the iso-likelihood shell around it. The volumes of these shells are used in the quadrature estimate of the Bayesian evidence described in @skillingNestedSampling2004.

# Statement of need

Nested sampling has grown immensely in popularity, due in part to its "black-box" nature, as well as the popularity of codes like MultiNest [@ferozMultimodalNestedSampling2008; @ferozMULTINESTEfficientRobust2009] in the astronomical community, where high-dimensional, multi-modal problems are commonplace. Recently, dynesty [@speagleDYNESTYDynamicNested2020] created an API which fully separates two independent steps of the nested sampling algorithm. These steps are, first: describing the statistical distribution of the live points, then second: likelihood-constrained sampling for replacing live points. This process is shown schematically in Figure 1.

NestedSamplers.jl mimics the API of dynesty by separating the independent steps of the nested sampling algorithm. Our library heavily utilizes multiple-dispatch enabled by the Julia programming language [@bezansonJuliaFreshApproach2017] to enable a highly-expressive, composable, and efficient nested sampling library. Multiple-dispatch allows fully encapsulating the independent components of the algorithm, in other words, the code specific to each bounding distribution and proposal algorithm is never repeated, and it comes with no performance loss thanks to the just-in-time compilation of Julia code. In addition, NestedSamplers.jl incorporates the AbstractMCMC.jl interface, which is an extensible interface for statistical sampling which enables entry points for using our nested samplers in various programming contexts.

NestedSamplers.jl currently has three bounding distributions: `NoBounds`, which represents the entire prior space [@skillingNestedSampling2004], `Ellipsoid`, which bounds the live points in an ellipsoid (equivalent to a multivariate Gaussian) [@mukherjeeNestedSamplingAlgorithm2006], and `MultiEllipsoid`, which uses an optimal clusterign of ellipsoids, first demonstrated by MultiNest [@ferozMultimodalNestedSampling2008; @ferozMULTINESTEfficientRobust2009]. In the future, we plan to implement the ball and cube distributions derived in @buchnerStatisticalTestNested2014; @buchnerCollaborativeNestedSampling2017.

NestedSamplers.jl has five restricted-likelihood sampling algorithms (proposal algorithms), using a slew of MCMC techniques. The first is the `Rejection` algorithm, which simply generates samples from the prior and rejects those outside the likelihood constraint [@skillingNestedSampling2004]. `RWalk` and `RStagger` use a Metropolis-Hastings-like walk for evolving a sample [@skillingNestedSamplingGeneral2006a]. `RSlice` and `Slice` use slice sampling in random [@handleyPolychordNestedSampling2015; @handleyPOLYCHORDNextgenerationNested2015] or principal directions [@nealSliceSampling2003], respectively, to evolve points. In the future, we plan to support Hamiltonian slice sampling [@speagleDYNESTYDynamicNested2020], which requires gradients and Jacobians.

NestedSamplers.jl currently uses a static nested sampler (integrator), where the number of live points is fixed throughout the sampling. Dynamic nested samplers allow tuning the integrator to avoid (or prefer) regions of high likelihood, which can be preferred in cases where posterior samples are more relevant than the Bayesian evidence estimate [@speagleDYNESTYDynamicNested2020]. In the future, we plan to implement a dynamic nested sampler which can make use of the existing bounding distributions and proposal algorithms.

# Comparisons to existing software

NestedSamplers.jl has many features similar to dynesty, and the predecessor code nestle. dynesty currently has more features, including a dynamic nested sampler, Hamiltonian slice sampling, and ball and cube distributions. The recent ultranest code [@buchnerNestedSamplingMethods2021] has a subset of the features of dynesty. MultiNest can be reproduced by using our `MultiEllipsoid` bounding distribution, and the exact algorithm can be reproduced with an appropriate `Rejection` proposal algorithm. The Polychord algorithm can be reproduced by using the `RSlice` algorithm, with typically an `Ellipsoid` bounding distribution.

# References