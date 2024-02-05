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
from dro_mev_functions.DRO_MEV_util import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class AsymmetricLogisticCopula():
    def __init__(self, alphas, thetas):
        self.m = alphas.shape[0]
        assert thetas.shape[0] == self.m, \
            'Number of alphas {} different from number of thetas {}'.format(self.m, thetas.shape[0])
        self.dim = thetas.shape[1]
        assert torch.all(thetas >= 0)
        if torch.any(thetas.sum(dim=0) != 1.):
            warn("thetas columns do not sum to 1, rescaling")
            thetas /= thetas.sum(dim=0, keepdim=True)
        self.alphas = alphas.view(1, -1, 1)
        self.thetas = thetas.unsqueeze(0)

    def sample(self, n_samples):
        Sm = rand_positive_stable(self.alphas, n_samples, self.m, 1)
        Wm = rand_exp(n_samples, self.m, self.dim)
        Xm = self.thetas * torch.where(self.alphas > 0, (Sm / Wm) ** self.alphas,
                                       torch.ones(1, 1, self.dim) / rand_exp(n_samples, self.m, 1))
        return Xm.max(dim=1)[0]

    def pickand(self, w):
        wtheta = w.unsqueeze(1) * self.thetas
        out_alpha_pos = torch.sum(wtheta ** (1. / self.alphas), dim=2, keepdim=True) ** self.alphas
        out_alpha_zero = torch.max(wtheta, dim=2, keepdim=True)[0]
        return torch.sum(torch.where(self.alphas > 0, out_alpha_pos, out_alpha_zero), dim=1).squeeze()

    

class SymmetricLogisticCopula():
    def __init__(self, dim, alpha):
        self.dim = dim
        self.alpha = alpha

    def sample(self, n_samples):
        if self.alpha > 0:
            S = rand_positive_stable(self.alpha, n_samples, 1)
            W = rand_exp(n_samples, self.dim)
            return (S / W) ** self.alpha
        else:
            return torch.ones(1, self.dim) / rand_exp(n_samples, 1)

    def pickand(self, w):
        if self.alpha > 0:
            return torch.sum(w ** (1 / self.alpha), dim=1) ** self.alpha
        else:
            return torch.max(w, dim=1)[0]

        
        
class stdfNSD(nn.Module):
    def __init__(self, alpha=torch.tensor([1.,2.]), rho=torch.tensor(-0.59)):
        super(stdfNSD, self).__init__()
        self.ndims = alpha.shape[0]
        self.alpha = alpha.detach().clone()
        self.rho   = rho.detach().clone()
        self.M     = self.sample_M(10000)
        
    def sample_M(self, n_samples):
        alpha = self.alpha
        rho   = self.rho
        gamma = tdist.gamma.Gamma(alpha, 1)
        D = gamma.sample((n_samples,))
        W = D**rho / (torch.lgamma(alpha+rho).exp() / torch.lgamma(alpha).exp())
        return W/W.mean(dim=0,keepdims=True)
    
    def sample(self, n_samples):
        ndims = self.ndims
        tries = 100

        P = 1./rand_exp(tries,n_samples).cumsum(axis=0)
        M = self.sample_M(tries*n_samples).view(tries,n_samples,ndims)
        U = torch.max(P[:,:,None]*M,dim=0)[0]
        U = torch.exp(-1./U)
        
        return U, M
    
    def forward(self,x):
        M = self.M
        ret = ((x[:,:,None].expand(-1,-1,1)*M.T[None,:,:].expand(1,-1,-1)).max(dim=1)[0]).mean(dim=1)
        return ret


    
    
class P0Module(nn.Module):
    def __init__(self, width, depth, d, d_z, act=nn.LeakyReLU()):
        super(P0Module, self).__init__()

        self.gen = nets.MLP(d_z, width, depth, d, act=act, bn=True)

    def forward(self, x):
        return sample_z(x.shape)

    def sample_z(self, shape):
        noise = torch.randn(shape)
        noise = noise.to(device)
        return l1_act(self.gen(noise))
    
    
class P0Module_pp(nn.Module):
    def __init__(self, width, depth, d, d_z, act=nn.LeakyReLU()):
        super(P0Module_pp, self).__init__()

        self.gen = nets.MLP(d_z, width, depth, d, act=act, bn=True)

    def forward(self, x):
        return sample_z(x.shape)

    def sample_z(self, shape):
        if len(shape) > 2:
            shape_p = (shape[0] * shape[1], shape[-1])
        else:
            shape_p = shape
        noise = torch.randn(shape_p)
        out = F.relu(self.gen(noise)).reshape(shape)
        return out

class DRONet(nn.Module):

    def __init__(self, c , l , 
            width : int, 
            depth : int, 
            d   : int, 
            d_z : int, 
            use_softmax : bool = False, 
            act = None, 
            model_type = 'evd', 
            init_net = None):
        '''
        c : a function that takes in two d-dim vectors and returns a scalar
        l : a function that takes in one d-dim vector and returns a scalar
        width : integer, number of points to use in the adversarial population
        d : integer, dimension of the vectors
        z_init : tensor, intialization of the adversarial  population
        '''
        super(DRONet, self).__init__()
        if init_net is not None:
            self.z_net = init_net.gen
        else:
            self.z_net = nets.MLP(d_z, width, depth, d, act=nn.LeakyReLU())

        self.act = act

        self.z_forward = lambda z: self.act(self.z_net(z)) + 1e-3

        self.c = c
        self.l = l

        self.use_softmax = use_softmax

        self.model_type = model_type

    def forward(self, x, lam, detach=False):
        noise = torch.randn_like(x).float() 
        Z = self.z_forward(noise)
        if detach:
            '''
            if 'evd' in self.model_type:
                Zc = Z.reshape(Nmax, -1, d).max(0)[0]
                xc = x.reshape(Nmax, -1, d).max(0)[0]
            elif self.model_type == 'unc':
            '''
            Zc = Z
            xc = x
            return (self.l(Z.detach()) - lam*self.c(Zc.detach(), xc)).max(0)[0]
        else:
            if self.use_softmax:
                val = (self.l(self.act(Z)) - lam*self.c(self.act(Z), x))
                amax = val.softmax(0)
                return (amax * val).sum()
            else:
                noise = torch.randn_like(x).float()
                Z = self.z_forward(noise)
                #Zc = Z.reshape(Nmax, -1, d).max(0)[0]
                #xc = x.reshape(Nmax, -1, d).max(0)[0]
                Zc = Z
                xc = x
                return (self.l(Z) - lam*self.c(Zc, xc)).max(0)[0]

    def expect(self, x, lam, detach=True):
        '''
        Computes the expectation with respect to a vector of x
        '''
        noise = torch.randn_like(x).float() 
        Z = self.z_forward(noise).float()
        if self.use_softmax:
            if detach:
                val = (self.l(self.act(Z.detach())) - lam * self.c(self.act(Z.detach()), x.unsqueeze(1)))
            else:
                val = (self.l(self.act(Z)) - lam.detach() * self.c(self.act(Z), x.unsqueeze(1)))
            amax = val.softmax(1)
            return torch.bmm(amax.unsqueeze(1), val.unsqueeze(2)).mean()
        else:
            '''
            if 'evd' in self.model_type:
                Zc = Z.reshape(Nmax, -1, d).max(0)[0]
                xc = x.reshape(Nmax, -1, d).max(0)[0]
            elif self.model_type == 'unc':
            '''
            Zc = Z.to(device)
            Z = Z.to(device)
            xc = x.to(device)
            lam = lam.to(device)
            if detach:
                return (self.l(Z.detach()) - lam * self.c(Zc.detach(), xc.unsqueeze(1))).max(1)[0].mean(0)
            else:
                #return (self.l(Z) - lam.detach() * self.c(Z, x.unsqueeze(1))).max(1)[0].mean(0)
                return (self.l(Z) - lam.detach() * self.c(Zc, xc.unsqueeze(1))).max(1)[0].mean(0)

    def sample_z(self, shape):
        noise = torch.randn(shape) 
        noise = noise.to(device)
        return self.z_forward(noise)
