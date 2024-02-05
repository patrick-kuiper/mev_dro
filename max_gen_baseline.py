import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import nets
import ot
import os

import numpy as np
import math

from tqdm import tqdm

import pickle

from dro_mev_functions.DRO_MEV_nn import *
from dro_mev_functions.DRO_MEV_train import *
from dro_mev_functions.DRO_MEV_util import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(12)
torch.manual_seed(12)


import yaml

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                    prog = 'MEVDRONet',
                    description = 'Computes adversarial risk for some data.')
    parser.add_argument('filename')
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        params = yaml.safe_load(f)

    d = params['d'] #2

    n_epochs = params['n_epochs'] #5001
    n_lam    = params['n_lam'] #20

    width  = params['width'] #500
    
    block_size = params['block_size']

    n_max = params['n_max'] #10
    rate = params['rate'] #10
    
    use_softmax = params['use_softmax'] #False
    experiment  = params['experiment'] #'evd'

    risk = params['risk'] #'cvar'

    n_eps  = params['n_eps'] #20
    n_runs = params['n_runs'] #1
    
    ########UPDATED 30JAN24: update loop vars############
    n_data_min = params['n_data_min']
    ##############################################################

    eps_max = params['eps_max'] #0.5

    data_file = params['data_file']
    cost_norm = params['cost_norm']

    fit_margins = params['fit_margins']

    gen_p0 = params['gen_p0']
    alpha  = params['alpha']
    ########UPDATED 27JAN24: SHORT PERIOD FOR TRAINING############
    baseline  = params['baseline']
    
    if baseline:
        data_file_short_period = params['data_file_short_period']
    ##############################################################        
    
    n_data = params['n_data'] #100

    try:
        eps_coef = params['eps_coef']
    except:
        eps_coef = None

    try:
        pretrain = params['pretrain']
    except:
        pretrain = False

    c = lambda z, x : (z - x).norm(cost_norm, dim=-1)

    synthetic_rate = params['synthetic_rate'] #0

    
    with open('{}.p'.format(data_file),'rb') as f:
        data = pickle.load(f)
        
    ########UPDATED 27JAN24: SHORT PERIOD FOR TRAINING############
    if baseline:
        with open('{}.p'.format(data_file_short_period),'rb') as f:
            data_short_period = pickle.load(f)
        n_data = data_short_period.shape[0]
    ##############################################################        

    if synthetic_rate:
        data_norm = data.norm(1,-1, keepdim=True)
        data_s = data / data_norm
        pp    = torch.tensor(np.random.exponential(synthetic_rate, size=(n_max, data.shape[0], 1)).cumsum(0))
        data  =  (data.repeat((n_max, 1, 1)) / pp).max(0)[0]

    if risk == 'cvar':
        true_risk = cvar(data.sum(-1)).mean().item()
    elif risk == 'cvar-min':
        true_risk = cvar_min(data.sum(-1)).mean().item()
    else:
        true_risk = data.norm(1, dim=-1).mean().item()

    print('True Risk: {}'.format(true_risk))

    losses = []
    try:
        if use_softmax:
            save_path  = data_file +'_{}_{}_softmax_gen_eps{}/'.format(experiment, risk, eps_max)
        else:
            if eps_coef is not None:
                save_path  = data_file +'_{}_{}_gen_eps{}_data{}_blocksize{}/'.format(experiment, risk, eps_coef, n_max, block_size, eps_coef)
            else:
                save_path  = data_file +'_{}_{}_gen_eps{}_data{}_blocksize{}/'.format(experiment, risk, eps_max, n_max, block_size)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
    except OSError as error:
        print(error)

    print('Saving in {}'.format(save_path))

    if data.shape[-1] == 2:
        plt.scatter(data[:,0], data[:,1], alpha=0.3)
        plt.title(data_file)
        plt.xlabel(r'$X_1$')
        plt.ylabel(r'$X_0$')
        plt.savefig(save_path + 'original_data.pdf')
        plt.close('all')

        cfg = nets.CFGEstimator(data)
        x_ = torch.linspace(0,1,100).cpu()
        x = torch.stack((x_, 1-x_),1).cpu()
        plt.plot(x_, cfg(x).cpu())
        plt.savefig(save_path + 'original_data_cfg.pdf')
        plt.close('all')

    if eps_max > 0:
        use_eps = True
        loop_vars = torch.linspace(n_data_min,eps_max,n_eps)

    else:
        use_eps = False
        
        loop_vars = torch.linspace(n_data_min, n_data, n_eps).int()

    n_loop = loop_vars.shape[0]

    adv_loss = torch.zeros((n_loop, n_runs))
    pop_loss = torch.zeros((n_loop, n_runs))

    adv_loss_mae = torch.zeros((n_loop, n_runs))
    pop_loss_mae = torch.zeros((n_loop, n_runs))

    adv_risk = torch.zeros((n_loop, n_runs))
    pop_risk = torch.zeros((n_loop, n_runs))
    true_risk_np = torch.zeros((n_loop, n_runs))

    E_p0 = torch.zeros((n_loop, n_runs))

    eps_N = torch.zeros(n_loop, n_runs)

    ep = []

    Fx = None

    for loop_idx, loop_var in enumerate(loop_vars):
        import ot
        #n_data = loop_var.int().item()

        for run in range(n_runs):

            if use_eps:
                # if we are looping over epsilon, set that as looping var
                eps    = loop_var
            else:
                n_data = loop_var.int().item()
            
            ########UPDATED 27JAN24: SHORT PERIOD FOR TRAINING############
            if baseline:
                sampled_data = data_short_period 
            else:
                sampled_data = data[np.random.choice(data.shape[0], n_data), :]
            ################################################################
            
            if block_size > 0:
                if block_size == 100:
                    sampled_data = sampled_data.reshape(5 * (1+loop_idx), -1, d).max(0)[0]
                else:
                    sampled_data = sampled_data.reshape(block_size, -1, d).max(0)[0]
                n_data = sampled_data.shape[0]
            
            sampled_data = sampled_data.to(device)
            cfg = nets.CFGEstimator(sampled_data)
            cfg = cfg.to(device)
            N_train = n_data

            spec = sampled_data / sampled_data.sum(-1, keepdim=True)
            spec = spec.to(device)
            pp = ( - torch.log(torch.rand(n_max, sampled_data.shape[0], 1))).cumsum(0)
            pp = pp.to(device)
            
            unit_margins = d * (spec.repeat((n_max, 1, 1)) / pp).max(0)[0]
            spec_samps = spec

            if fit_margins:
                from scipy.stats import invweibull, genextreme

                margin_params = torch.zeros(d, 3) 
                Fx = torch.zeros(sampled_data.shape)

                margin_errors = 0
                qq_x = torch.linspace(0, 1, n_data)
                plt.plot(qq_x, qq_x, label='Reference', color='red')
                for idx in range(d):
                    try:
                        select_sample = sampled_data[:,idx].cpu()
                        param = torch.tensor(genextreme.fit(select_sample))
                    except ValueError:
                        print('Raised Value Error, aborting.')
                        margin_errors+=1

                    margin_params[idx, :] = param  # shape, loc, scale
                    print('Shape, margin {} : {}'.format(idx, param[0]))
                    Fx[:,idx] = torch.tensor(genextreme.cdf(sampled_data[:,idx].cpu(), *param))
                    plt.plot(qq_x, Fx[:,idx].sort()[0], label='Margin CDF {}'.format(idx))
                plt.legend()
                plt.savefig(save_path + 'qq_margin.pdf')
                plt.close('all')

                if gen_p0:
                    net_p0 = P0Module(32, 2, d, d, act=nn.LeakyReLU())  # fit the P0 network
                    Fx = Fx.to(device)
                    net_p0 = net_p0.to(device)
                    cfg = cfg.to(device)
                    fit_p0(net_p0, Fx, save_path, a=cfg)
                    net_p0.eval()

                    with torch.no_grad():
                        spec_samps   = net_p0.sample_z(( N_train, d)).detach()

                if margin_errors > 0:
                    print('Obtained {} margin errors, aborting run.'.format(margin_errors))
                    adv_risk[loop_idx, run] = torch.nan
                    pop_risk[loop_idx, run] = torch.nan

                    adv_loss[loop_idx, run] = torch.nan
                    pop_loss[loop_idx, run] = torch.nan

                    adv_loss_mae[loop_idx, run] = torch.nan
                    pop_loss_mae[loop_idx, run] = torch.nan

                    losses.append((loop_var, 0, 0))
                    break

                unit_margins = (d * spec_samps.repeat((n_max, 1, 1)) / pp).max(0)[0] 
                margin_params = margin_params.to(device)
                unit_margins = unit_margins.to(device)
                X = margin_params[:,2] * (unit_margins ** margin_params[:,0] - 1) / margin_params[:,0]  + margin_params[:,1]
            else:
                X = unit_margins

            if not use_eps:
                # compute the W distance between X and the data

                if eps_coef is None:
                    M = ot.dist(X[:n_data], sampled_data, metric='cityblock')
                    a, b = torch.ones(sampled_data.shape[0],) / n_data, torch.ones(sampled_data.shape[0],) / n_data
                    eps = ot.emd(a, b, M, log=True)[1]['cost'] #/ np.sqrt(n_data) 
                    if 'evd' in experiment:
                        if gen_p0:
                            M = ot.dist(spec_samps[:n_data], spec, metric='cityblock')
                            a, b = torch.ones(sampled_data.shape[0],) / n_data, torch.ones(sampled_data.shape[0],) / n_data
                            eps = ot.emd(a, b, M, log=True)[1]['cost'] 
                        else:
                            eps = eps / sampled_data.sum(-1).mean() / d
                    eps_N[loop_idx, run] = eps 
                else:
                    eps = eps_coef / np.sqrt(n_data)
                print('Epsilon = {} for N = {}:'.format(eps, n_data))

            if d == 2:
                if gen_p0:
                    with torch.no_grad():
                        p0_samps = d * net_p0.sample_z((1000,2)).detach().cpu()
                        plt.scatter(p0_samps[:,0], p0_samps[:,1], alpha=0.3)
                        plt.savefig(save_path + 'p0.pdf')
                        plt.close('all')

                plt.scatter(sampled_data[:,0].cpu(), sampled_data[:,1].cpu(), alpha=0.3, label='Original')
                plt.scatter(X[:,0].cpu(), X[:,1].cpu(), alpha=0.3, label='P_0')
                plt.title(data_file)
                plt.xlabel(r'$X_1$')
                plt.ylabel(r'$X_0$')
                plt.legend()
                plt.savefig(save_path + 'data_overlay.pdf')
                plt.close('all')

            if 'evd' in experiment:
                # If we are using the constrained case, we only want to compute distances between the spectrals

                if risk == 'cvar':
                    def l(z):
                        m = d * (z.repeat((n_max, 1, 1)) / pp).max(0)[0]
                        if fit_margins:
                            m = margin_params[:,2] * (m ** margin_params[:,0] - 1) / margin_params[:,0] + margin_params[:,1]
                        else:
                            m = m
                        return cvar(m.sum(-1,keepdims=True), alpha)
                elif risk == 'cvar-min':
                    def l(z, plot=False):
                        m = d * (z.repeat((n_max, 1, 1)) / pp).max(0)[0]
                        if fit_margins:
                            m = margin_params[:,2] * (m ** margin_params[:,0] - 1) / margin_params[:,0] + margin_params[:,1]
                        else:
                            m = m
                        if plot:
                            plt.scatter(m[:,0], m[:,1])
                            plt.savefig(save_path + 'p_adv_full.pdf')
                            plt.close('all')
                        return cvar_min(m.sum(-1,keepdims=True), alpha)
                else:
                    def l(z):
                        m = d * (z.repeat((n_max, 1, 1)) / pp).max(0)[0]
                        if fit_margins:
                            m = margin_params[:,2] * (m ** margin_params[:,0] - 1) / margin_params[:,0] + margin_params[:,1]
                        else:
                            m = m
                        return  m.norm(1, dim=-1)

                E_p0[loop_idx, run] = (X).sum(-1).mean(0)
                X = spec_samps

            else:
                # Fit the original model as the decomposition max R x \theta where \theta is spectral measure
                if risk == 'cvar':
                    def l(z):
                        return cvar(z.sum(-1,keepdims=True), alpha)
                elif risk == 'cvar-min':
                    def l(z):
                        return cvar_min(z.sum(-1,keepdims=True), alpha)
                else:
                    def l(z):
                        return z.norm(1, dim=-1)
                E_p0[loop_idx, run] = X.sum(-1).mean(0)

            print('Epsilon   = {}'.format(eps))
            print('Risk P0   = {}'.format(l(X).mean()))
            print('E[|P0|_1] = {}'.format(E_p0[loop_idx, run]))

            print('========= Data stats ========')

            print('Max P0:   {}'.format(X.max().item()))
            print('Max data: {}'.format(data.max().item()))

            print('Mean P0:   {}'.format(X.mean().item()))
            print('Mean data: {}'.format(data.mean().item()))

            if 'evd' in experiment:
                if eps == 0:
                    X = X.to(device)
                    pop, adv = train(X, eps, c, l, act=relu_act, n_epochs=2 * n_epochs, save_path=save_path, Fx=Fx, pretrain=pretrain, experiment=experiment, init_p=net_p0 if gen_p0 else None, use_softmax = use_softmax, n_lam = n_lam) #increase epochs for 0
                else:
                    X = X.to(device)
                    pop, adv = train(X, eps, c, l, act=relu_act, n_epochs=n_epochs, save_path=save_path, Fx=Fx, pretrain=pretrain, experiment=experiment, init_p=net_p0 if gen_p0 else None, use_softmax = use_softmax, n_lam = n_lam)
            else:
                X = X.to(device)
                pop, adv = train(X, eps, c, l, act=id_act, n_epochs=n_epochs, save_path=save_path, experiment=experiment, use_softmax = use_softmax, n_lam = n_lam)

            mse_pop = (pop - true_risk)**2 
            mse_adv = (adv - true_risk)**2 

            mae_pop = true_risk - pop 
            mae_adv = true_risk - adv

            true_risk_np[loop_idx, run] = true_risk
            adv_risk[loop_idx, run] = adv
            pop_risk[loop_idx, run] = pop

            adv_loss[loop_idx, run] = mse_adv
            pop_loss[loop_idx, run] = mse_pop

            adv_loss_mae[loop_idx, run] = mae_adv
            pop_loss_mae[loop_idx, run] = mae_pop

        losses.append((loop_var, mse_pop, mse_adv))

    plt.plot(loop_vars,eps_N.mean(1))
    plt.fill_between(loop_vars,eps_N.mean(1) - eps_N.std(1), eps_N.std(1) + eps_N.mean(1), alpha=0.3)
    plt.xlabel(r'$N$')
    plt.ylabel(r'$\delta$')
    plt.savefig('eps_n.pdf')
    plt.close('all')

    pickle_dict = {'losses' : losses,
            'E_p0' : E_p0,
            'true_risk' : true_risk_np,
            'p0_risk' : pop_risk,
            'adv_risk': adv_risk,
            'pop_loss': pop_loss, 
            'adv_loss': adv_loss, 
            'pop_loss_mae': pop_loss_mae,
            'adv_loss_mae': adv_loss_mae}

    with open(save_path + '/stats{}_{}.p'.format(eps_max, n_data), 'wb') as f:
        pickle.dump(pickle_dict, f)
        
    losses = np.array(losses)

    plt.plot(losses[:,0], pop_loss.mean(1), label=r'$\mathbb{P}_0$')
    plt.fill_between(losses[:,0], pop_loss.mean(1) - pop_loss.std(1), pop_loss.mean(1) + pop_loss.std(1), alpha=0.3)

    plt.plot(losses[:,0], adv_loss.mean(1), label=r'$\mathbb{P}_\star$')
    plt.fill_between(losses[:,0], adv_loss.mean(1) - adv_loss.std(1), adv_loss.mean(1) + adv_loss.std(1), alpha=0.3)

    if use_eps:
        plt.xlabel(r'$\delta$')
    else:
        plt.xlabel(r'$N$')

    plt.ylabel(r'$(E_{X\sim archimax} [\ell(X)] - E_{X\sim P}[\ell(X)])^2$')
    #plt.title(r'$N={}$'.format(n_data))
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}/err_vs_eps_{}_n={}_std_sm={}_eps={}.pdf'.format(save_path, experiment, n_data, use_softmax, eps_max))
    plt.close('all')

    plt.plot(losses[:,0], pop_loss_mae.mean(1), label=r'$\mathbb{P}_0$')
    plt.fill_between(losses[:,0], pop_loss_mae.mean(1) - pop_loss_mae.std(1), pop_loss_mae.mean(1) + pop_loss_mae.std(1), alpha=0.3)

    plt.plot(losses[:,0], adv_loss_mae.mean(1), label=r'$\mathbb{P}_\star$')
    plt.fill_between(losses[:,0], adv_loss_mae.mean(1) - adv_loss_mae.std(1), adv_loss_mae.mean(1) + adv_loss_mae.std(1), alpha=0.3)

    if use_eps:
        plt.xlabel(r'$\delta$')
    else:
        plt.xlabel(r'$N$')
    plt.ylabel(r'$|E_{X\sim archimax} [\ell(X)] - E_{X\sim P}[\ell(X)]|$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}/err_vs_eps_{}_n={}_std_sm={}_eps={}_mae.pdf'.format(save_path, experiment, n_data, use_softmax, eps_max))
    plt.close('all')

