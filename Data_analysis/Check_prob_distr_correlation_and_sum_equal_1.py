import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
import time
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
import scipy.io
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import lognorm
import math
import itertools
import random
import scipy.io
import time


fig_width = 15
fig_height =10
fig_size =  [fig_width,fig_height]
params = {'backend': 'TkAgg',
    'axes.labelsize': 40,
    #'text.fontsize': 30,
    'axes.titlesize':40,
    'legend.fontsize': 30,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'text.usetex': False,
    'font.family': 'sans-serif',
    'figure.figsize': fig_size,
    'font.size': 30}
rcParams.update(params)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


# this file is to check that the chosen prob distr ensures that pairwise correlations are always c and the sparsity is gamma for any pattern group size

Tot_patterns = 2.*10**4
c = 0.04
gamma = 0.002
C = (c - gamma)/(1. - gamma)
pmax = 100

def p1(a, b):
    return a * b + (1. - a) * (1. - b)
def p11(a, b):
    return a * b**2 + (1. - a) * (1. - b)**2
def p110(a, b):
    return a * b**2*(1. -b) + (1. - a) * b* (1. - b)**2
def p111(a, b):
    return a * b**3 + (1. - a) * (1. - b)**3
def p100(a, b):
    return a * b*(1. - b)**2 + (1. - a) * b**2 * (1. - b)

def get_alpha_beta(gamma, C_hat):
    beta_roots = np.roots([ 2.,  -3., (1 + 2 * gamma * (1 - gamma) * (1 - C_hat)), - gamma * (1 - gamma) * (1 - C_hat)])
    alpha_candidate = 0.
    beta_candidate = 0.
    for beta in beta_roots:
        if beta.imag == 0 and beta.real >= 0 and beta.real <= 1.000000001:
            beta = beta.real
            alpha = (gamma + beta - 1)/(2 * beta - 1)
            C_hat_recomputed=(p11(alpha, beta) - gamma**2)/(gamma*(1-gamma))
            gamma_recomputed = p1(alpha, beta)
            if alpha >= 0 and alpha <= 1.000000001 and abs(C_hat_recomputed - C_hat) < 0.001 and abs(gamma_recomputed - gamma) < 0.001:
                
                #if alpha_candidate != 0 :  # i.e. if \alpha_candidate is already assigned
                #    print("Using second values of alpha and beta. ")
                alpha_candidate = alpha
                beta_candidate = beta
    return alpha_candidate, beta_candidate

def prob_parent_pattern(p, n1): # p = number correlated patterns
    alpha, beta = get_alpha_beta(gamma, C)
    n0 = p - n1
    prefactor = float(math.factorial(p))/float((math.factorial(n1)*math.factorial(p-n1)))
    #print n1, n0, prefactor
    prob = prefactor*(alpha * beta**n1 * (1-beta)**n0 + (1. - alpha)*(1. - beta)**n1 * beta**n0)
    return prob

def prob_random_pattern(p ,n1): # p = number correlated patterns
    n0 = p - n1
    prefactor =  float(math.factorial(p))/float((math.factorial(n1)*math.factorial(p-n1)))
    #print 'p ', p, 'n1 ', n1, 'p - n1', n0, 'prefactor', prefactor,' (c**n1 * (1. - c)**n0) ', (c**n1 * (1. - c)**n0)
    prob = 0
    if n1 == 0:
        #prob = prefactor * ((1. - gamma) - gamma * (1. - c)**(p-1))
        prob = prefactor * ((1. - gamma) - gamma * (1. - c)*(p-1))
    else:
        prob = prefactor * gamma * c**(n1 -1)*(1. - c )**(p - (n1-1))
    return prob


  
alpha, beta = get_alpha_beta(gamma, C)
print( c , " = ", (p110(alpha, beta) + p111(alpha, beta))/(2*p110(alpha, beta)+ p111(alpha, beta)+ p100(alpha, beta)))
print( c , " = ", (gamma*c*(1.-c) +gamma*c**2)/(2*gamma*c*(1.-c) + gamma*(1.-c)**2 + gamma*c**2) )  # this works only for gamma = c
#print c , " = ", (prob_random_pattern(3 ,2) +prob_random_pattern(3 ,3))/(2*prob_random_pattern(3 ,2) +prob_random_pattern(3 ,3) + prob_random_pattern(3 ,1)) #no, because of the prefactor

n = 20
'''
lst = list(itertools.product([0, 1], repeat=n))
lst = np.flip(lst, axis = 0)
n_combinations = len(lst)
print( 'n_combinations', n_combinations)
'''

sum1 = 0
sum2 = 0
idx = 0
for j in range(n+1):
    #print j
    sum1 += prob_random_pattern(n ,j)
    sum2 += prob_parent_pattern(n ,j)
    idx += 1
print( sum1, sum2)

