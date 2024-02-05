import torch
import torch.nn as nn
import torch.distributions as tdist
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import math
import nets
import numpy as np
from tqdm import tqdm


def rand_exp(*dims):
    return -torch.rand(*dims).log()

def rand_simplex(batch_size, dim):
    exp = rand_exp(batch_size, dim)
    return exp / torch.sum(exp, dim=1, keepdim=True)

def rand_positive_stable(alpha, *dims):
    U = math.pi*torch.rand(*dims)
    W = rand_exp(*dims)

    return (torch.sin(alpha * U) / (U.sin() ** (1 / alpha))) * (torch.sin((1-alpha)*U) / W) ** (1/alpha - 1)

def rand_sym_log(n_samples, dim, alpha):
    if alpha > 0:
        S = rand_positive_stable(alpha, n_samples, 1)
        W = rand_exp(n_samples, dim)
        return (S / W) ** alpha
    else:
        return torch.ones(1, dim) / rand_exp(n_samples, 1)


def l1norm(x):
    return x / x.norm(1,-1, keepdim=True)

def sample_mixture(comp, probs, rates, N, Nmax=10):

    multi = tdist.Categorical(probs) # construct a categorial dist
    cats  = multi.sample((N,1))      # sample the catgories

    samplers = [comp[i] for i in cats] # choose the sampler from each category

    samps_unit = torch.stack([l1norm(d.sample(Nmax)) for d in samplers])  # project to simplex

    sampler_rates = torch.stack([rates[i] for i in cats])  # get the shape parameter for each

    pp = torch.stack([rand_exp((Nmax,1))  for r in sampler_rates]) # sample the point process

    samps = samps_unit.shape[-1] * (pp * samps_unit).max(1)[0]  # unit frechet samples
    samps = (samps ** sampler_rates - 1) / sampler_rates        # samples with different scales

    return samps


def sample_z(self, shape):
        noise = torch.randn(shape) 
        return self.z_forward(noise)


def l1_act(z):
    return z.abs() / z.abs().sum(-1,keepdims=True)

def sm_act(z):
    return torch.softmax(z,-1)

def relu_act(z):
    return F.relu(z)

def id_act(z):
    return z

def ht_act(z):
    return F.hardtanh(z, 0, 1.1)

def cvar(X, alpha=0.85):
    '''
    Expects a 1-d r.v. with shape N x 1
    '''
    N = X.shape[0]
    X_sort  = X.sort(0)[0]
    x_alpha = X_sort[math.floor(N * (alpha))]

    #cdf_xalpha = (X <= x_alpha).float().mean()

    es = 1 / alpha * X * (X <= x_alpha) #* N / (X <= x_alpha).sum()

    return es.squeeze(-1)

def cvar_min(X, alpha=0.95):
    '''
    Expects a 1-d r.v. with shape N x 1
    '''
    N = X.shape[0]
    X_sort  = X.sort(0)[0]
    var = (X_sort[math.floor(N * alpha)]) - 1e-3

    es = F.softplus(X - var) / ( 1 - alpha ) + var
    return es.squeeze(-1)


def var(X, alpha=0.95):
    N = X.shape[0]
    X_sort  = X.sort(0)[0]
    x_alpha = X_sort[int(N * alpha)]

    return -x_alpha

