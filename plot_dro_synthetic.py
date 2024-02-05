import pickle
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser(
                prog = 'MEVDRONet',
                description = 'Computes adversarial risk for some data.')
parser.add_argument('filename')
args = parser.parse_args()

with open(args.filename, 'r') as f:
    params = yaml.safe_load(f)

d = params['d']

n_epochs = params['n_epochs']
n_lam    = params['n_lam'] 

width  = params['width'] 
n_data = params['n_data']
block_size = params['block_size']

if block_size > 0:
    n_data = n_data // block_size
    if block_size == 100:
        n_data = 20

n_max = params['n_max'] 
rate = params['rate'] 

use_softmax = params['use_softmax'] 
experiment  = params['experiment'] 

risk = params['risk'] 

n_eps  = params['n_eps'] 
n_runs = params['n_runs'] 

eps_max = params['eps_max'] 

data_file = params['data_file'] 
cost_norm = params['cost_norm']

try:
    eps_coef = params['eps_coef']
except:
    eps_coef = None

synthetic_rate = params['synthetic_rate'] #0

import os

if eps_coef is not None:
    save_path_unc     = data_file + '_unc_{}_gen_eps{}_data{}_blocksize{}/'.format(risk, eps_coef, n_max, block_size)
    save_path_evd     = data_file + '_evd_{}_gen_eps{}_data{}_blocksize{}/'.format(risk, eps_coef, n_max, block_size)
    save_path_evd_sm  = data_file + '_evd-sm_{}_gen_eps{}_data{}_blocksize{}/'.format(risk, eps_coef, n_max, block_size)
else:
    save_path_unc     = data_file + '_unc_{}_gen_eps{}_data{}_blocksize{}/'.format(risk, eps_max, n_max, block_size)
    save_path_evd     = data_file + '_evd_{}_gen_eps{}_data{}_blocksize{}/'.format(risk, eps_max, n_max, block_size)
    save_path_evd_sm  = data_file + '_evd-sm_{}_gen_eps{}_data{}_blocksize{}/'.format(risk, eps_max, n_max, block_size)

true_key = 'true_risk'
pop_key  = 'p0_risk'
adv_key  = 'adv_risk'

print(save_path_unc)

try:
    with open(save_path_evd + 'stats{}_{}.p'.format(eps_max, n_data), 'rb') as f:
        data = pickle.load(f)
    true = data[true_key]
    pop  = data[pop_key]
    adv  = data[adv_key]
    adv = adv * (adv < 1e3)
    x = np.stack(data['losses'])[:,0]
except :
    adv = None
    print('EVD stats not found, skipping')

try:
    print(save_path_unc + 'stats0_{}.p'.format(n_data))
    with open(save_path_unc + 'stats{}_{}.p'.format(eps_max, n_data), 'rb') as f:
        datau = pickle.load(f)
    true = datau[true_key]
    pop  = datau[pop_key]
    advu = datau[adv_key]
    advu = advu * (advu.abs() < 1e3)
    x = np.stack(datau['losses'])[:,0]
except :
    advu = None
    print('Unconstrained stats not found, skipping')

try:
    with open(save_path_evd_sm + 'stats{}_{}.p'.format(eps_max, n_data), 'rb') as f:
        datasm = pickle.load(f)
    true = datasm[true_key]
    pop  = datasm[pop_key]
    advsm = datasm[adv_key]
    advsm = advsm * (advsm < 1e3)
    x = np.stack(datasm['losses'])[:,0]
except :
    advsm = None
    print('EVD-sm stats not found, skipping')


start_domain = 0
x_domain = (x / true.mean(-1))
x_domain_max = np.around(torch.max(x / true.mean(-1)).detach().numpy())
x_ticks = np.arange(x_domain_max + 1, dtype=int)

error_P0 = torch.abs(torch.tensor(np.nanmean(pop.numpy(), -1)) - true.mean(-1))
error_P0_lower = np.log(error_P0) - (np.nanstd(error_P0.numpy(), -1) / error_P0)
error_P0_upper = np.log(error_P0) + (np.nanstd(error_P0.numpy(), -1) / error_P0)


fig, ax = plt.subplots()
ax.plot(x_domain, torch.log(error_P0), color = "red", marker ='x', label=r'$P_0$: Non-DRO EVD Risk')
ax.fill_between(x_domain, error_P0_lower, error_P0_upper, color = "red", alpha=0.3)

if adv is not None:
    error_Pstar = torch.tensor(np.nanmean(adv.numpy(), -1)) - true.mean(-1)
    error_Pstar[0] = error_P0[0]
    error_Pstar_lower = np.log(error_Pstar) - (np.nanstd(error_Pstar.numpy(), -1) / error_Pstar)
    error_Pstar_upper = np.log(error_Pstar) + (np.nanstd(error_Pstar.numpy(), -1) / error_Pstar)
    ax.plot(x_domain, torch.log(error_Pstar), color = "blue", marker ='*', label=r'$P_\star$: DRO EVD Risk')
    ax.fill_between(x_domain, error_Pstar_lower, error_Pstar_upper, color = "blue", alpha=0.3)
    
if advsm is not None:
    error_PstarUni = torch.tensor(np.nanmean(advsm.numpy(), -1)) - true.mean(-1)
    error_PstarUni[0] = error_P0[0]
    error_PstarUni_lower = np.log(error_PstarUni) - (np.nanstd(error_PstarUni.numpy(), -1) / error_PstarUni)
    error_PstarUni_upper = np.log(error_PstarUni) + (np.nanstd(error_PstarUni.numpy(), -1) / error_PstarUni)
    ax.plot(x_domain, torch.log(error_PstarUni), color = "orange", marker ='*', label='$P_\star (\mathbb{E}[w_i] \approx \frac{1}{d})$: DRO EVD\nRisk w/ Uniform Margins')
    ax.fill_between(x_domain, error_PstarUni_lower, error_PstarUni_upper, color = "orange", alpha=0.3)
    
if advu is not None:
    error_Punc = torch.tensor(np.nanmean(advu.numpy(), -1)) - true.mean(-1)
    error_Punc[0] = error_P0[0]
    error_Punc_lower = np.log(error_Punc) - (np.nanstd(error_Punc.numpy(), -1) / error_Punc)
    error_Punc_upper = np.log(error_Punc) + (np.nanstd(error_Punc.numpy(), -1) / error_Punc)
    ax.plot(x_domain, torch.log(error_Punc), color = "green", marker ='*', label=r'$P_\star (unc)$: Non-MEV Risk')
#     error_Punc_lower[0]  =  error_PstarUni_lower[0]
#     error_Punc_upper[0]  =  error_PstarUni_upper[0]
    error_Punc_lower[0]  =  error_Pstar_lower[0]
    error_Punc_upper[0]  =  error_Pstar_lower[0]
    ax.fill_between(x_domain, error_Punc_lower, error_Punc_upper, color = "green", alpha=0.3)

y_domain_max = np.around(torch.max(error_Pstar).detach().numpy())
print(y_domain_max)
y_ticks = np.arange(y_domain_max + 1, dtype=int)
ax.spines['top'].set_color('none')
ax.spines['right'].set_position(('axes', 1.0))
ax.spines['left'].set_color('none')
ax.spines['bottom'].set_position(('axes', 0.0))
plt.xticks(x_ticks)
ax.yaxis.tick_right()
plt.xlabel(r'$\delta$ Normalized by True Risk', fontsize = 18)
ax.yaxis.set_label_position("right")
plt.ylabel(r'$Log \vert \mathbb{E}[\ell(X_{P_{true}})] - \mathbb{E}[\ell(X_{P_{model}})] \vert$', fontsize = 18)
plt.title("ASL Expected Risk Evaluated Over\nIncreasing Uncertainty ($\delta$)", fontsize = 20)
plt.legend(loc= "lower right", fontsize = 13)
plt.tight_layout()
plt.savefig('raw_comparison_{}_N_{}_{}.pdf'.format(data_file, n_data, risk))
