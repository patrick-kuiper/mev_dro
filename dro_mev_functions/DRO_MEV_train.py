import torch
import torch.nn as nn
import torch.distributions as tdist
import torch.nn.functional as F
import torch.optim as optim

import math
import nets
import numpy as np
from tqdm import tqdm

from dro_mev_functions.DRO_MEV_nn import *
from dro_mev_functions.DRO_MEV_util import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(X, eps, c, l, act, n_epochs, save_path, two_opt=True, Fx=None, pretrain=False, cfg=None, experiment=None, init_p=None, use_softmax = False, n_lam = 10):
    
    X = X.to(device)
    
    if 'unc' in save_path:
        lam = torch.ones(1, requires_grad=True) # init lambda
    else:
        lam = torch.ones(1, requires_grad=True) # init lambda

    lam.requires_grad=True
    d = X.shape[-1]
    
    if 'unc' in save_path:
        rn = DRONet(c, l, width=16, depth=1, d=d, d_z=d, use_softmax=use_softmax, act=act, model_type=experiment, init_net=init_p)
        rn = rn.to(device)
        rn.train()
    else:
        rn = DRONet(c, l, width=16, depth=1, d=d, d_z=d, use_softmax=use_softmax, act=act, model_type=experiment, init_net=init_p)
        rn = rn.to(device)
        rn.train()
        for module in rn.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                rn.eval()


    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
            0.1 * averaged_model_parameter + 0.9 * model_parameter

    swa_model = torch.optim.swa_utils.AveragedModel(rn, avg_fn=ema_avg)

    if two_opt:
        if 'unc' in save_path:
            print('using larger learning rate')
            #opt_net = optim.AdamW((rn.parameters()),  lr=3e-4, betas=[0.5,0.999])#, eps=1e-1)
            opt_net = optim.AdamW((rn.parameters()),  lr=5e-5)#, betas=[0.5,0.999])#, eps=1e-1)
            opt_l   = optim.AdamW([lam], lr=1e-4)#, betas=[0.5,0.999]) #1e-2
        else:
            #opt_net = optim.AdamW((rn.parameters()),  lr=1e-4, betas=[0.5,0.999])#, eps=1e-3)
            opt_net = optim.AdamW((rn.parameters()),  lr=1e-5)
            #opt_l   = optim.AdamW([lam], lr=1e-4, betas=[0.5,0.999]) #1e-2
            opt_l   = optim.AdamW([lam], lr=2e-5) #1e-2
    else:
        opt_net = optim.Adam(list(rn.parameters()) + [lam], lr=1e-3, betas=[0.5, 0.999])
        opt_l   = opt_net
    scheduler = optim.lr_scheduler.ExponentialLR(opt_net, gamma=0.9998)

    loss_max = []
    loss_min = []

    stop_crit = 0

    for epoch in range(n_epochs):
        for _ in range(n_lam):
            # iterate over the lambda
            opt_l.zero_grad()
            #opt_net.zero_grad()

            risk = rn.expect(X, lam.abs())
            risk = risk.to(device)
            eps = eps.to(device)
            lam = lam.to(device)
            loss_lam = eps * lam.abs() + risk

            loss_lam.backward()
            #opt_net.step()
            opt_l.step()
        loss_min.append(loss_lam.item())

        opt_net.zero_grad()
        loss = 0 
        loss = -rn.expect(X, lam.abs(), detach=False)

        if 'evd-sm' in save_path:
#             in_noise = torch.randn(1000,2)
######ADDED 18JAN24 to make multi-dim#########
            in_noise = torch.randn(1000,d)
            in_noise = in_noise.to(device)
##############################################
            loss += 100 * F.l1_loss(rn.z_net(in_noise).mean(0), torch.ones(d).to(device) / d)
        loss_max.append(loss.item())
        loss.backward()

        opt_net.step()
        scheduler.step()

        if epoch % 100 == 0:
            
            swa_model.update_parameters(rn)

            risk = swa_model.module.expect(X, lam.abs())
            loss_lam = eps * lam.abs() + risk

            print(epoch)
            print('lambda = {}'.format(lam.abs().item()))
            print('p0  risk = {}'.format(l(X).mean().item()))
            print('adv risk = {}'.format(loss_lam.item()))

            plt.scatter(X[:,0].cpu(), X[:,1].cpu(), label=r'$\mathbb{P}_0$', alpha=0.5)
            
            ac = rn.sample_z(X.shape).detach()
            print('E Margins {}'.format(ac.mean(0).cpu().numpy()))

            plt.scatter(ac[:,0].cpu(), ac[:,1].cpu(), label=r'$\mathbb{P}_\star$', alpha=0.5)
            plt.legend()
            plt.savefig('{}/scatter.pdf'.format(save_path))
            plt.close('all')

            plt.plot(loss_max, label=r'$\mathcal{L}_\max$')
            plt.plot(loss_min, label=r'$\mathcal{L}_\lambda$')
            plt.legend()
            plt.savefig('{}/losses.pdf'.format(save_path))
            plt.close('all')

    return l(X).mean().item(), loss_lam.item()

def train_pp(p0_net, v_n, a_n, x, Y_n, p0, eps, act, n_epochs, save_path, two_opt=True, Fx=None, pretrain=False, cfg=None, experiment=None, init_p=None):

    print('Delta = {}'.format(eps))

    lam = torch.tensor(1/(eps+1e-3)*np.ones(1), requires_grad=True) # init lambda

    lam.requires_grad=True

    opt_l   = optim.AdamW([lam], lr=1e-1)
    scheduler = optim.lr_scheduler.ExponentialLR(opt_l, gamma=0.9998)

    loss_max = []
    loss_min = []

    stop_crit = 0

    p0 = 1 - ( -F.relu(v_n).mean() ).exp()

    print('p0 {} : '.format(p0.item()))

    N_E   = 1000
    n_max = 100
    d = Y_n.shape[-1]

    for epoch in range(n_epochs):
        opt_l.zero_grad()

        a_n = (-torch.rand((N_E, n_max)).log()).cumsum(-1)
        with torch.no_grad():
            Y_n = p0_net.sample_z(( N_E, n_max, d )).detach()

        v_n = (Y_n * x).max(-1)[0]
        E_0_l = F.relu( 1 - lam.abs() * ( F.relu(a_n - v_n) ).min(1)[0] ).mean(0) 
        loss = lam.abs() * eps + E_0_l

        loss.backward()
        opt_l.step()

        if epoch % 100 == 0:

            print('Epoch number: {}'.format(epoch))
            print('lambda = {}'.format(lam.abs().item()))
            p0 = 1 - (-F.relu(v_n)).mean().exp()
            print('p0  risk = {}'.format(p0.item()))
            print('adv risk = {}'.format(loss.item()))
            adv2 = 1 - (-F.relu((x * Y_n - 1 / (x * lam.abs())).max(-1)[0]).mean()).exp()
            print('adv2 risk = {}'.format(adv2.item()))

    return p0.item(), loss.item(), adv2.item()

def fit_p0(model, Fx, save_path, a=None):
    print('Initializing P_\star as P0...')
    Fx = Fx.to(device)
    model = model.to(device)
    model.train()
    opt   = optim.AdamW(model.parameters(), lr=1e-2)
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9998)

    d = Fx.shape[-1]
    N_w = Fx.shape[0]  
    N_s = 1000

    epochs = 100

    with tqdm(range(epochs)) as tepoch:
        for _ in tepoch:

            opt.zero_grad()

            w = rand_simplex(N_w, d)
            w = w.to(device)
            
            shape = (N_s, d)
            s = (model.sample_z(shape)).unsqueeze(1)
            s = s.to(device)
            
            Aw = d * ( (w * s).max(-1)[0] ).mean(0)
            Aw = Aw.to(device)
            
            reg = F.l1_loss(s.squeeze(1).mean(0), torch.ones_like(s.squeeze(1).mean(0)) / d)
            reg = reg.to(device)
            
            
            if a is not None:
                a = a.to(device)
                Aw = Aw.log().to(device)
                aw = a(w).log()
                aw = aw.to(device)
                loss = F.l1_loss(Aw, aw) + reg
            else:
                zw = ( -Fx.log() / w ).min(-1)[0]

                nll = (Aw * zw - Aw.log()).mean()

                loss = nll #+ 100*reg

            tepoch.set_postfix(loss=loss.item())

            loss.backward()
            opt.step()
            sched.step()

    print('Final loss: {}'.format(loss.item()))
    print(s.mean(0))

    x_ = torch.linspace(0, 1, 100)
#     x  = torch.stack([x_, 1-x_],-1)
######ADDED 18JAN24 to make multi-dim#########
    x_grid  = create_grid_array(100,d)
    x = torch.tensor(x_grid).to(device)
###############################################
    aw = (d*((x * s).max(-1)[0]).mean(0)).cpu().detach().numpy()

    plt.plot(x_, aw)
    plt.savefig(save_path + 'fitted_pick.pdf')
    plt.close('all')
    
    
def create_grid_array(rows, cols):
    """Creates an array with decimal values between 0 and 1, rows summing to 1,
     using a grid-like pattern and without randomness.
    """

    # Create a base grid with evenly spaced values from 0 to 1 along each row
    grid = np.linspace(0, 1, cols, endpoint=False).reshape(1, cols)
    grid = np.repeat(grid, rows, axis=0)

    # Add a linearly increasing offset to each row, ensuring non-randomness
    offset = np.linspace(0, 1, rows, endpoint=False).reshape(-1, 1)
    offset = np.repeat(offset, cols, axis=1)
    grid += offset

    # Wrap around values greater than 1 to create a seamless grid
    grid %= 1

    # Normalize each row to ensure they sum to 1
    grid /= np.sum(grid, axis=1, keepdims=True)

    return grid


def fit_p0_pp(model, X, save_path):
    print('Initializing P_\star as P0...')

    model.train()
    opt   = optim.AdamW(model.parameters(), lr=1e-2)
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9998)

    d = X.shape[-1]
    N_w = X.shape[0]  
    N_s = 1000

    epochs = 50

    with tqdm(range(epochs)) as tepoch:
        for _ in tepoch:

            opt.zero_grad()

            shape = (N_s, d)

            #x = -torch.rand(N_w, d).log()
            x = torch.rand(N_w, d)

            zx = (X * x).min(-1)[0]

            Y = (model.sample_z(shape)).unsqueeze(1)  # sample Y

            Ax = ( (x * Y).max(-1)[0] ).mean(0)       # expectation of max transform

            nll = (Ax * zx - Ax.log()).mean()

            loss = nll 

            tepoch.set_postfix(loss=loss.item())

            loss.backward()
            opt.step()
            sched.step()

    print('Final loss: {}'.format(loss.item()))
    print(Y.mean(0))


