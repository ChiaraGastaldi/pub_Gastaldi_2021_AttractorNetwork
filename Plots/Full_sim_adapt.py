import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
import time
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

fig_width = 12
fig_height = 8
fig_size =  [fig_width,fig_height]
params = {'backend': 'TkAgg',
    'axes.labelsize': 30,
    #'text.fontsize': 30,
    'axes.titlesize':30,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'text.usetex': False,
    'font.family': 'sans-serif',
    'figure.figsize': fig_size,
    'font.size': 30}
rcParams.update(params)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

gamma = 0.002
N = 10000
P = 16
tau = 1
h = 0.1
load = P/N
C_hat = 0.2
A = 1.
h0 = 0
rm = 1.
b = 100.
t_max = 500

#t_vect=np.arange(0,t_max-h,h)
#_distr_gains
overlap_t = np.loadtxt("../Output_full_sim_adapt/adapt_overlap_time_neus%s_gamma%s_P%s_tau%s_dt%s_chat%s_A%s_rm%s_b%s.dat"%(N, gamma, P,  tau, h, C_hat, A, rm ,b), unpack=True)


plt.figure()
"""
gs = gridspec.GridSpec(1,1, width_ratios=[1],height_ratios=[1],left=0.15, bottom=0.3, right=0.95, top=0.90, wspace=0.3, hspace=0.5)
plt.close('all')
ax0 = plt.subplot(gs[0])
"""
for i in range(1,P+1):
    #ax0.plot(overlap_t[0,:],overlap_t[i,:], label = i)
    plt.plot(overlap_t[0,:],overlap_t[i,:], label = i)
#ax0.plot(overlap_t[0,:],overlap_t[-1,:], 'k--', alpha = 0.5, label = "J0")
#ax0.plot([0, t_max], [C_hat, C_hat], 'k--', alpha = 0.5)
#ax0.set_xlabel('Time [AU]')
#ax0.set_ylabel('Overlaps')
#ax0.set_ylim([-0.1, 1.1])
#ax0.set_xlim([0, t_max])
plt.xlabel('Time [s]')
plt.ylabel('Overlaps')
plt.ylim([-0.1, 1.1])
plt.xlim([0, t_max])
#plt.legend()
#_distr_gains
plt.savefig("../Figures_full_sim_adapt/full_sim_neus%s_gamma%s_P%s_A%s_rm%s_h0%s_b%s_tau%s_h%s_C_hat%s_tmax%s.pdf"%(N, gamma, P, A, rm, h0, b, tau, h,  C_hat, t_max),   transparent=True)
plt.show()


""""
## Intro plot

fig_width = 6
fig_height = 3
fig_size =  [fig_width,fig_height]
params = {'backend': 'TkAgg',
    'axes.labelsize': 30,
    'text.fontsize': 30,
    'axes.titlesize':30,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'text.usetex': False,
    'font.family': 'sans-serif',
    'figure.figsize': fig_size,
    'font.size': 30}
rcParams.update(params)
#rcParams = plt.PyDict(PyPlot.matplotlib."rcParams")
#rc("svg", fonttype="none")
plt.rcParams['svg.fonttype'] = 'path'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

gamma = 0.1
N = 10000
P = 16
tau = 1
h = 0.1
load = P/N
C_hat = 0.
A = 1
h0 = 0
rm = 1
t_max = 200

#t_vect=np.arange(0,t_max-h,h)

overlap_t = np.loadtxt("output_full_adapt/adapt_overlap_time_neus%s_gamma%s_P%s_tau%s_dt%s_chat%s.dat"%(N, gamma, P,  tau, h, C_hat), unpack=True)


plt.figure()
gs = gridspec.GridSpec(1,1, width_ratios=[1],height_ratios=[1],left=0.15, bottom=0.3, right=0.95, top=0.90, wspace=0.3, hspace=0.5)
plt.close('all')
ax0 = plt.subplot(gs[0])

for i in range(1,P):
    ax0.plot(overlap_t[0,:],overlap_t[i,:], label = i)
#ax0.plot(overlap_t[0,:],overlap_t[-1,:], 'k--', alpha = 0.5, label = "J0")
#ax0.plot([0, t_max], [C_hat, C_hat], 'k--', alpha = 0.5)
ax0.set_xlabel('Time [AU]')
#ax0.set_ylabel('Overlaps')
ax0.set_ylim([-0.1, 1.1])
ax0.set_xlim([0, t_max])
#plt.legend()

plt.savefig("figures_full_adapt/INTRO_full_sim_neus%s_gamma%s_P%s_A%s_rm%s_h0%s_tau%s_h%s_C_hat%s_tmax%s.pdf"%(N, gamma, P, A, rm, h0, tau, h,  C_hat, t_max),   transparent=True)
plt.show()
"""
