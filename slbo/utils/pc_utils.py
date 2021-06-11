import numpy as np
import copy
import os
import scipy.spatial
import scipy.signal


def median_trick(X, args):
    #median trick for computing the bandwith for kernel regression.
    N = X.shape[0]
    #print(X.shape)
    perm = np.random.choice(N, np.min([N,args.update_size * args.buffer_width]), replace=False)
    dsample = X[perm]
    pd = scipy.spatial.distance.pdist(dsample)
    sigma = np.median(pd)
    return sigma

def compute_cov_pi(phi):
    #cov = np.zeros((phi.shape[1],phi.shape[1]))

    #for i in range(len(phi)):
    #    cov += np.outer(phi[i],phi[i])
    cov = np.dot(phi.T,phi)
    cov /= phi.shape[0]

    #print(cov)

    return cov

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]




def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


