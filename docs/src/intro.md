# Introduction

Nested sampling is a statistical technique first described in Skilling (2004)[^1] as a method for estimating the Bayesian evidence. Conveniently, it also produces samples with importance weighting proportional to the posterior distribution. To understand what this means, we need to comprehend [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem).

## Bayes' theorem

Bayes' theorem, in our nomenclature, is described as the relationship between the *prior*, the *likelihood*, the *evidence*, and the *posterior*. In its entirety-

```math
p(\theta | x) = \frac{p(x | \theta)p(\theta)}{p(x)}
```

### Posterior

 ``p(\theta | x)`` - the probability of the model parameters (``\theta``) conditioned on the data (``x``)

### Likelihood

``p(x | \theta)`` - the probability of the data (``x``) conditioned on the model parameters (``\theta``)

### Prior

``p(\theta)`` - the probability of the model parameters

### Evidence

``p(x)`` - the probability of the data

If you are familiar with Bayesian statistics and Markov Chain Monte Carlo (MCMC) techniques, you should be somewhat familiar with the relationships between the posterior, the likelihood, and the prior. The evidence, though, is somewhat hard to describe; what does "the probability of the data" mean? Well, another way of writing the evidence, is this integral

```math
p(x)  \equiv Z = \int_\Omega{p(x | \theta) \mathrm{d}\theta}
```

which is like saying "the likelihood of the data [``p(x | \theta)``] integrated over *all of parameter space* [``\Omega``]". We have to write the probability this way, because the data are statistically dependent on the model parameters. This integral is intractable for all but the simplest combinations of distributions ([conjugate distributions](https://en.wikipedia.org/wiki/Conjugate_prior)), and therefore it must be estimated or approximated in some way.

## What can we do with the evidence?

Before we get into approximating the Bayesian evidence, let's talk about why it's important. After all, for most MCMC applications it is simply a normalization factor to be ignored (how convenient!).

## Further reading

For further reading, I recommend reading the cited sources in the footnotes, as well as the references below

* [dynesty documentation](https://dynesty.readthedocs.io)


[^1]: Skilling 2004