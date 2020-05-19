import torch
from torch.distributions import HalfCauchy, HalfNormal

import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm

from pyro.distributions import *


import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, Predictive

assert pyro.__version__.startswith('1')
pyro.enable_validation(True)       # can help with debugging
pyro.set_rng_seed(0)

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

# alpha = 0.1
# model(data)

optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO())


def train(num_iterations):
    losses = []
    pyro.clear_param_store()
    for j in tqdm(range(num_iterations)):
        loss = svi.step(data, num_particles=10)
        losses.append(loss)

    return losses


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


alpha = 1
elbo = train(25000)

plt.figure()
plt.plot(elbo)
plt.show()

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