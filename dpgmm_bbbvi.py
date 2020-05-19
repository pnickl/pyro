import os
from collections import defaultdict
from functools import partial

import numpy as np
import pyro
import pyro.distributions as dist
import scipy.stats
import torch
import torch.distributions.constraints as constraints
from torch.distributions import HalfCauchy, HalfNormal
import torch.nn.functional as F
# from matplotlib import pyplot
import matplotlib.pyplot as plt
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
from pyro.poutine import block, replay, trace
from pyro.distributions import *

from tqdm import tqdm

n_steps = 12000
pyro.set_rng_seed(2)

# enable validation (e.g. validate parameters of distributions)
pyro.enable_validation(True)

# clear the param store in case we're in a REPL
pyro.clear_param_store()

data = torch.cat((MultivariateNormal(-8 * torch.ones(2), torch.eye(2)).sample([50]),
                  MultivariateNormal(8 * torch.ones(2), torch.eye(2)).sample([50]),
                  MultivariateNormal(torch.tensor([-0.5, 1]), torch.eye(2)).sample([50])))

N = data.shape[0]

def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

def model(data, **kwargs):
    with pyro.plate("beta_plate", T - 1):
        beta = pyro.sample("beta", Beta(1, alpha))

    with pyro.plate("var_plate", T * 2):
        var = pyro.sample("var", HalfNormal(scale=0.5 * torch.ones(1)))

    with pyro.plate("corr_plate", T):
        corr = pyro.sample("corr", LKJCorrCholesky(d=2, eta=1e6 * torch.ones(1)).expand([T]))

    with pyro.plate("mu_plate", T):
        L_sigma = torch.bmm(torch.diag_embed(torch.sqrt(var.view(T, 2))), corr)
        mu = pyro.sample("mu", MultivariateNormal(torch.zeros(2), scale_tril=L_sigma))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(mix_weights(beta)))
        pyro.sample("obs", MultivariateNormal(mu[z], scale_tril=L_sigma[z]), obs=data)

def guide(data, **kwargs):
    gamma = pyro.param('gamma', alpha * torch.ones(T - 1,), constraint=constraints.positive)

    zeta = pyro.param('zeta', lambda: Uniform(0.25, 0.5).sample([T * 2]))

    psi = pyro.param('psi', lambda: Uniform(1e6, 1e7).sample(), constraint=constraints.positive)

    tau = pyro.param('tau', lambda: MultivariateNormal(torch.zeros(2), 5 * torch.eye(2)).sample([T]))
    pi = pyro.param('pi', torch.ones(N, T) / T, constraint=constraints.simplex)

    with pyro.plate("beta_plate", T - 1):
        q_beta = pyro.sample("beta", Beta(torch.ones(T - 1), gamma))

    with pyro.plate("var_plate", T * 2):
        q_var = pyro.sample("var", HalfNormal(scale=zeta))

    with pyro.plate("corr_plate", T):
        q_corr = pyro.sample("corr", LKJCorrCholesky(d=2, eta=psi).expand([T]))

    with pyro.plate("mu_plate", T):
        q_L_sigma = torch.bmm(torch.diag_embed(torch.sqrt(q_var.view(T, 2))), q_corr)
        q_mu = pyro.sample("mu", MultivariateNormal(tau, scale_tril=q_L_sigma))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(pi))

T = 3
alpha = 1

def relbo(model, guide, *args, **kwargs):
    approximation = kwargs.pop('approximation')

    # We first compute the elbo, but record a guide trace for use below.
    traced_guide = trace(guide)
    elbo = pyro.infer.Trace_ELBO(max_plate_nesting=1)
    loss_fn = elbo.differentiable_loss(model, traced_guide, *args, **kwargs)

    # We do not want to update parameters of previously fitted components
    # and thus block all parameters in the approximation apart from z.
    guide_trace = traced_guide.trace
    replayed_approximation = trace(replay(block(approximation, expose=['z']), guide_trace))
    approximation_trace = replayed_approximation.get_trace(*args, **kwargs)

    relbo = -loss_fn - approximation_trace.log_prob_sum()

    # By convention, the negative (R)ELBO is returned.
    return -relbo


def approximation(data, components, weights):
    assignment = pyro.sample('assignment', dist.Categorical(weights))
    result = components[assignment](data)
    return result

def truncate(alpha, centers, vars, corrs, weights):
    threshold = alpha**-1 / 100.
    true_centers = centers[weights > threshold]

    vars = vars.view(T, 2)
    true_vars = vars[weights > threshold]

    true_corrs = corrs[weights > threshold, ...]

    _sigmas = torch.bmm(true_vars.sqrt().view(-1, 2).diag_embed(), true_corrs)
    true_sigmas = torch.zeros(len(_sigmas), 2, 2)
    for n in range(len(_sigmas)):
        true_sigmas[n, ...] = torch.mm(_sigmas[n, ...], _sigmas[n, ...].T)

    true_weights = weights[weights > threshold] / torch.sum(weights[weights > threshold])
    return true_centers, true_sigmas, true_weights

def boosting_bbvi():
    # T=2
    n_iterations = 3
    initial_approximation = partial(guide, index=0)
    components = [initial_approximation]
    weights = torch.tensor([1.])
    wrapped_approximation = partial(approximation, components=components, weights=weights)

    locs = [0]
    scales = [0]

    for t in range(1, n_iterations + 1):

        # Create guide that only takes data as argument
        wrapped_guide = partial(guide, index=t)
        losses = []

        adam_params = {"lr": 0.01, "betas": (0.90, 0.999)}
        optimizer = Adam(adam_params)

        # Pass our custom RELBO to SVI as the loss function.
        svi = SVI(model, wrapped_guide, optimizer, loss=relbo)
        for step in tqdm(range(n_steps)):
            # Pass the existing approximation to SVI.
            loss = svi.step(data, approximation=wrapped_approximation)
            losses.append(loss)

            if step % 100 == 0:
                print('.', end=' ')

        # Update the list of approximation components.
        components.append(wrapped_guide)

        # Set new mixture weight.
        new_weight = 2 / (t + 1)

        # # In this specific case, we set the mixture weight of the second component to 0.5.
        # if t == 2:
        #     new_weight = 0.5
        weights = weights * (1-new_weight)
        weights = torch.cat((weights, torch.tensor([new_weight])))

        # Update the approximation
        wrapped_approximation = partial(approximation, components=components, weights=weights)

    # We make a point-estimate of our model parameters using the
    # posterior means of tau and phi for the centers and weights
    posterior_predictive = Predictive(guide, num_samples=100)
    posterior_samples = posterior_predictive.forward(data)

    mu_mean = posterior_samples['mu'].detach().mean(dim=0)
    var_mean = posterior_samples['var'].detach().mean(dim=0)
    corr_mean = posterior_samples['corr'].detach().mean(dim=0)
    beta_mean = posterior_samples['beta'].detach().mean(dim=0)

    weights_mean = mix_weights(beta_mean)

    centers, sigmas, weights = truncate(alpha, mu_mean, var_mean, corr_mean, weights_mean)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)

    plt.scatter(data[:, 0], data[:, 1], color="blue", marker="+")
    plt.scatter(centers[:, 0], centers[:, 1], color="red")

    from math import pi

    t = torch.arange(0, 2 * pi, 0.01)
    circle = torch.stack([torch.sin(t), torch.cos(t)], dim=0)

    for n in range(len(sigmas)):
        ellipse = torch.mm(torch.cholesky(sigmas[n, ...]), circle)
        plt.plot(ellipse[0, :] + centers[n, 0], ellipse[1, :] + centers[n, 1],
                 linestyle='-', linewidth=2, color='g', alpha=1.)
    plt.show()

if __name__ == '__main__':
    boosting_bbvi()