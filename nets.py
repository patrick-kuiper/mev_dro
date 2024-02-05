import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
import scipy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(net, init_dict, gain=1, input_class=None):
    def init_func(m):
        if input_class is None or type(m) == input_class:
            for key, value in init_dict.items():
                param = getattr(m, key, None)
                if param is not None:
                    if value == 'normal':
                        nn.init.normal_(param.data, 0.0, gain)
                    elif value == 'xavier':
                        nn.init.xavier_normal_(param.data, gain=gain)
                    elif value == 'kaiming':
                        nn.init.kaiming_normal_(param.data, a=0, mode='fan_in')
                    elif value == 'orthogonal':
                        nn.init.orthogonal_(param.data, gain=gain)
                    elif value == 'uniform':
                        nn.init.uniform_(param.data)
                    elif value == 'zeros':
                        nn.init.zeros_(param.data)
                    elif value == 'very_small':
                        nn.init.constant_(param.data, 1e-3*gain)
                    elif value == 'xavier1D':
                        nn.init.normal_(param.data, 0.0, gain/param.numel().sqrt())
                    elif value == 'identity':
                        nn.init.eye_(param.data)
                    else:
                        raise NotImplementedError('initialization method [%s] is not implemented' % value)
#activation functions
class quadratic(nn.Module):
    def __init__(self):
        super(quadratic,self).__init__()

    def forward(self,x):
        return x**2

class quadratic(nn.Module):
    def __init__(self):
        super(quadratic,self).__init__()

    def forward(self,x):
        return x*F.relu(x)

class cos(nn.Module):
    def __init__(self):
        super(cos,self).__init__()

    def forward(self,x):
        return torch.cos(x)

class sin(nn.Module):
    def __init__(self):
        super(sin,self).__init__()

    def forward(self,x):
        return torch.sin(x)

class swish(nn.Module):
    def __init__(self):
        super(swish,self).__init__()

    def forward(self,x):
        return torch.sigmoid(x)*x

class relu2(nn.Module):
    def __init__(self,order=2):
        super(relu2,self).__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.order = order

    def forward(self,x):
        #return F.relu(self.a.to(x.device)*x)**(self.order)
        return F.relu(x)**(self.order)

class leakyrelu2(nn.Module):
    def __init__(self,order=2):
        super(leakyrelu2,self).__init__()
        self.a = nn.Parameter(torch.ones(1))
        #self.a = torch.ones(1)
        self.order = order

    def forward(self,x):
        return F.leaky_relu(self.a.to(x.device)*x)**self.order

class mod_softplus(nn.Module):
    def __init__(self):
        super(mod_softplus,self).__init__()

    def forward(self,x):
        return F.softplus(x) + x/2 - torch.log(torch.ones(1)*2).to(device=x.device)

class mod_softplus2(nn.Module):
    def __init__(self):
        super(mod_softplus2,self).__init__()

    def forward(self,x,d):
        return d*(1+d)*(2*F.softplus(x) - x  - 2*torch.log(torch.ones(1)*2).to(device=x.device))

class mod_softplus3(nn.Module):
    def __init__(self):
        super(mod_softplus3,self).__init__()

    def forward(self,x):
        return F.relu(x) + F.softplus(-torch.abs(x)) 

class swish(nn.Module):
    def __init__(self):
        super(swish,self).__init__()

    def forward(self,x):
        return x*torch.sigmoid(x) 

class soft2(nn.Module):
    def __init__(self):
        super(soft2,self).__init__()

    def forward(self,x):
        return torch.sqrt(x**2 + 1) / 2 + x / 2

class soft3(nn.Module):
    def __init__(self):
        super(soft3,self).__init__()

    def forward(self,x):
        return torch.logsigmoid(-x) 
class Shallow(nn.Module):
    def __init__(self,input_size,out_size):
        super(Shallow, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size,input_size),quadratic(),nn.Linear(input_size,out_size))

    def forward(self,x):
        return self.net(x)

class PositiveLinear(nn.Linear):
    def __init__(self, **args):
        super(PositiveLinear, self).__init__()


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, layers, out_size, act=nn.LeakyReLU(), bn=True, bias=False, use_pe=0):
        super(MLP, self).__init__()

        if use_pe:
            self.pe = PositionalEncodingLayer(L=use_pe)
            self.fc1 = nn.Linear(2 * use_pe, hidden_size, bias=False)
        else:
            self.pe = None
            self.fc1 = nn.Linear(input_size, hidden_size, bias=False)

        if bn:
            self.bn = nn.BatchNorm1d(hidden_size, track_running_stats=True)
            #self.bn = nn.InstanceNorm1d(hidden_size)
        else:
            self.bn = None
        mid_list = []
        for i in range(layers):
            if bn:
                mid_list += [nn.Linear(hidden_size,hidden_size), nn.BatchNorm1d(hidden_size, track_running_stats=True), act]
                #mid_list += [nn.Linear(hidden_size,hidden_size), nn.InstanceNorm1d(hidden_size), act]
            else:
                mid_list += [nn.Linear(hidden_size,hidden_size, bias=False), act]
        self.mid = nn.Sequential(*mid_list)
        self.out = nn.Linear(hidden_size, out_size, bias=bias)
        self.act = act
        init_weights(self, {'weights':'xavier', 'bias':'zeros'})

    def forward(self,x):
        if self.pe is not None:
            out = self.pe(x)
        else: 
            out = x
        out = self.fc1(out)
        if self.bn:
            out = self.bn(out)
        out = self.act(out)
        out = self.mid(out)
        out = self.out(out)#.clamp(max=0.1)
        #out = 0.5*torch.tanh(out)
        #out = -x ** 2 + 5
        #out = -10*x  + 5
        #out = 10*torch.exp(-x) -8*x
        return out

from scipy.interpolate import interp1d
import math

class PositionalEncodingLayer(nn.Module):
    def __init__(self, L=20, device='cpu'):
        super(PositionalEncodingLayer, self).__init__()
        scale1 = 2**torch.arange(0, L)*math.pi
        scale2 = 2**torch.arange(0, L)*math.pi + math.pi 
        self.scale = torch.stack((scale1,scale2),1).view(1,-1).to(device)

    def forward(self, x):
        xs = list(x.shape)
        vs = xs[:-1] + [-1]
        return torch.sin(x.unsqueeze(-1) @ self.scale).view(*vs)

class HistoryMLP(MLP):
    def __init__(self, t_history, history, input_size, hidden_size, layers, out_size, act=nn.LeakyReLU(), bn=False, bias=False, in_x=True, use_hist=True, pe=False):
        self.use_hist = use_hist
        if use_hist:
            self.interpolator = interp1d(t_history, history)
        if pe:
            super(HistoryMLP, self).__init__(2 * pe + int(use_hist) + int(in_x), hidden_size, layers, out_size, act, bn, bias)
            self.t_pe = PositionalEncodingLayer(L=pe)
        else:
            super(HistoryMLP, self).__init__(input_size + int(use_hist) + int(in_x), hidden_size, layers, out_size, act, bn, bias)
            self.t_pe = None

    def forward(self, x, t):
        if x is not None:
            if self.use_hist:
                hist = torch.tensor(self.interpolator(t)).float()
                x_in = torch.cat([x,hist,t], -1)
            else:
                x_in = torch.cat([x,t], -1)
                if self.t_pe is not None:
                    x_in = torch.cat([x,self.t_pe(t)], -1)
        else:
            if self.use_hist:
                hist = torch.tensor(self.interpolator(t)).float()
                x_in = torch.cat([hist,t], -1)
            else:
                x_in = t
                if self.t_pe is not None:
                    x_in = self.t_pe(t)
        return super().forward(x_in)

class SplineRegression(torch.nn.Module):
    def __init__(
            self,
            input_range,
            order=3,
            knots=10):
        super(SplineRegression, self).__init__()
        if isinstance(knots, int):
            knots = np.linspace(input_range[0], input_range[1], knots)
        num_knots = len(knots)

        knots = np.hstack([knots[0]*np.ones(order),
                           knots,
                           knots[-1]*np.ones(order)])
        self.basis_funcs = scipy.interpolate.BSpline(
            knots, np.eye(num_knots+order-1), k=order)
        self.linear = torch.nn.Linear(num_knots+order-1, 1)

        x = np.linspace(input_range[0], input_range[1], 100)
        y = self.basis_funcs(x)
        #print(y.shape)
        #plt.plot(x, y)
        #plt.show()

    def forward(self, x):
        x_shape = x.shape
        x_basis = self.basis_funcs(x.reshape(-1))
        x_basis = torch.from_numpy(x_basis).float()
        out = self.linear(x_basis)
        return out.reshape(x_shape)


class MetaCE(nn.Module):
    def __init__(self, samples, est_F=None, survival=False):
        super(MetaCE, self).__init__()
        (self.n_samples, self.dim) = samples.shape
        self.samples = samples.T.unsqueeze(0)
        order = torch.argsort(self.samples, dim=2, descending=False)
        F_ = torch.argsort(order, dim=2)
        if est_F is None:
            est_F = 'n+1'
        try:
            est_F = float(est_F)
        except ValueError:
            if est_F == 'n+1':
                self.F = (F_ + 1) / (self.n_samples + 1)
            else:
                self.F = est_F.T
        else:
            assert 0 <= est_F <= 1
            self.F = (F_ + est_F) / self.n_samples
        self.survival = survival
        if self.survival:
            self.F = 1 - self.F

    def est_survival(self, CDF):
        '''
        estimator : a subclass of metace
        theshold : d dimensional vector of thresholds
        Given an estimator compute survival prob from threshold
        '''
        assert self.survival, 'must be survival copula'

        t = (1-CDF).log()
        w = t / t.sum()
        A = self(w)
        survival = (t.sum()*A).exp()
        return survival

class CFGEstimator(MetaCE):
    def __init__(self, samples, lambda_fun=None, est_F=None, survival=False):
        super(CFGEstimator, self).__init__(samples, est_F, survival)
        self.logF = (self.F + 1e-6).log()
        
        self.logF = self.logF.to(device)
        xi_ek = (-self.logF / torch.eye(self.dim).unsqueeze(2).to(device)).min(dim=1)[0]
        self.loghA_ek = -(xi_ek + 1e-6).log().mean(dim=1)
        if lambda_fun is None:
            lambda_fun = lambda x: x
        self.lambda_fun = lambda_fun

    def forward(self, w):
        w = w.to(device)
        xi = (-self.logF/w.unsqueeze(2)).min(dim=1)[0] + 1e-6
        hA = (-xi.log().mean(dim=1) - self.lambda_fun(w) @ self.loghA_ek).exp()
        return hA
