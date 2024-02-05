import torch
import torch.nn as nn
import torch.distributions as tdist

import matplotlib.pyplot as plt
import math
import sys
import pickle

from dro_mev_functions.DRO_MEV_nn import *
from dro_mev_functions.DRO_MEV_train import *
from dro_mev_functions.DRO_MEV_util import *


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise Exception("Must input either: 'asl' or 'sl' ")
    
    data_gen_type = sys.argv[1]
    
    N = 10000
    nsd = stdfNSD(torch.tensor([3.0, 3.0]), torch.tensor(2.1))
    x = nsd.M
    plt.scatter(x[:,0], x[:,1])
    plt.savefig('nsd.pdf')
    
    if data_gen_type == "asl":
        ########## Asymmetric Logistic ##############
        print('=====> Generating Asymmetric Logistic Mixture')
        rates = torch.tensor([0.0001, 0.9])
        d = 2


        alpha = 0.8 * torch.ones(1)
        alphas = torch.tensor((0.5 * torch.ones_like(alpha), alpha))

        thetas = torch.rand(d)
        thetas = torch.stack((thetas, 1 - thetas), dim=0)

        as1 = AsymmetricLogisticCopula(alphas, thetas)

        alpha2 = 0.1 * torch.ones(1)
        alphas2 = torch.tensor((0.5 * torch.ones_like(alpha), alpha))

        thetas2 = 0.01 * torch.rand(d)
        thetas2 = torch.stack((thetas2, 1 - thetas2), dim=0)

        as2 = AsymmetricLogisticCopula(alphas2, thetas2)

        dists = [as1, as2]
        probs = torch.tensor([0.90, 0.1])

        s = sample_mixture(dists, probs, rates, N, 100)
        print(s)
        with open('mixture_asl_asl.p','wb') as f:
            pickle.dump(s, f)

        print('Max:{}'.format(s.max().item()))
        
    else:
        ########## Symmetric Logistic ##############
        print('=====> Generating Symmetric Logistic Mixture')

        rates = torch.tensor([0.01, 0.8])
        s1 = SymmetricLogisticCopula(2, 0.95)
        s2 = SymmetricLogisticCopula(2, 0)
        dists = [s1, s2]
        probs = torch.tensor([0.95, 0.05])
        
        s = sample_mixture(dists, probs, rates, N, 100)

        with open('mixture_sl_sl.p','wb') as f:
            import pickle
            pickle.dump(s, f)
        print('Max:{}'.format(s.max().item()))
