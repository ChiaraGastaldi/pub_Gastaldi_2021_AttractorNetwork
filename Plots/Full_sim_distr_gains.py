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

fig_width = 13
fig_height =10
fig_size =  [fig_width,fig_height]
params = {'backend': 'TkAgg',
    'axes.labelsize': 40,
    #'text.fontsize': 30,
    'axes.titlesize':40,
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

gamma = a = 0.002
N = 10000
P = 5000
A = 1.
beta = 100.
h0 = 0.25
tau = 1
h = 0.1
t_max = 15
t_onset = 0.5
t_offset = 8
load = P/N
resolution = 200
resolution_factor = 1
b = beta
rm = 1.

linewidth= 2
color_m1_MF =   '#194A8D'
color_m2_MF =  '#AF601A'

t_vect=np.arange(0.,t_max,h)


plt.figure()
gs = gridspec.GridSpec(4,1, width_ratios=[1],height_ratios=[1,1,1,1],left=0.23, bottom=0.12, right=0.97, top=0.9, wspace=0.1, hspace=0.1)
plt.close('all')
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])

#### graph 0 ####

C_hat=0.
ax0.set_title("Overlaps' dynamics")

data = np.loadtxt("../Output_full_sim_distr_gains/overlap_time_neus%s_gamma%s_P%s_tau%s_dt%s_chat%s_A%s_b%s.dat"%(N, gamma, P, tau, h, C_hat, A,  b), unpack=True)
#time, m1, m2, stab = np.loadtxt("../../AttractorNetwork/output_2pop/2pop_with_noise_vs_time_gamma%s_time_step%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_Chat%s.dat"%(gamma, h, resolution, resolution_factor, rm, A, h0, b, load, C_hat), unpack = True, skiprows=1, delimiter = ",")
for i in range(1,P+1):
    ax0.plot(data[0,:],data[i,:], lw=linewidth)
ax0.plot(data[0,:],data[1,:], 'k--', lw=linewidth, alpha = 0.5, label = r'$\sigma$ FS')
print( "final sigma", data[1,-1])

ax0.plot([t_onset,t_onset], [-0.1, 1.1], "k:", alpha=0.5)
ax0.plot([t_offset,t_offset], [-0.1, 1.1], "k:", alpha=0.5)

#ax0.plot(time, m1, '--', lw = 2, c = color_m1_MF)
#ax0.plot(time, m2, '--',  lw = 2, c = color_m2_MF)

ax0.set_xlim([0,t_max])
ax0.set_ylim([-0.1,1.1])
ax0.plot([0, t_max], [0.0141230,0.0141230], 'b:', label = r'$\sigma$ MF')
#ax0.set_ylabel("$\hat{C}=%s$"%C_hat,labelpad=90, rotation=0, x=-1)
ax0.set_ylabel("Overlaps")
ax0.axes.get_xaxis().set_visible(False)



C_hat=0.15
data = np.loadtxt("../Output_full_sim_distr_gains/overlap_time_neus%s_gamma%s_P%s_tau%s_dt%s_chat%s_A%s_b%s.dat"%(N, gamma, P, tau, h, C_hat, A,  b), unpack=True)
#time, m1, m2, stab = np.loadtxt("../../AttractorNetwork/output_2pop/2pop_with_noise_vs_time_gamma%s_time_step%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_Chat%s.dat"%(gamma, h, resolution, resolution_factor, rm, A, h0, b, load, C_hat), unpack = True, skiprows=1, delimiter = ",")
for i in range(1,P+1):
    ax1.plot(data[0,:],data[i,:], lw=linewidth)
ax1.plot([t_onset,t_onset], [-0.1, 1.1], "k:", alpha=0.5)
ax1.plot([t_offset,t_offset], [-0.1, 1.1], "k:", alpha=0.5)

#ax1.plot(time, m1, '--', lw = 2, c = color_m1_MF)
#ax1.plot(time, m2, '--',  lw = 2, c = color_m2_MF)

ax1.set_xlim([0,t_max])
#ax1.set_ylim([-0.1,1.1])
#ax1.set_ylabel("$\hat{C}=%s$"%C_hat,labelpad=90, rotation=0)
ax1.set_ylabel("Overlaps")
ax1.axes.get_xaxis().set_visible(False)

"""
C_hat=0.2
data = np.loadtxt("../Output_full_sim/overlap_time_neus%s_gamma%s_P%s_tau%s_dt%s_chat%s_A%s_rm%s_b%s.dat"%(N, gamma, P, tau, h, C_hat, A, rm, b), unpack=True)
time, m1, m2, stab = np.loadtxt("../../AttractorNetwork/output_2pop/2pop_with_noise_vs_time_gamma%s_time_step%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_Chat%s.dat"%(gamma, h, resolution, resolution_factor, rm, A, h0, b, load, C_hat), unpack = True, skiprows=1, delimiter = ",")
for i in range(1,P+1):
    ax2.plot(data[0,:],data[i,:], lw=linewidth)
ax2.plot([t_onset,t_onset], [-0.1, 1.1], "k:", alpha=0.5)
ax2.plot([t_offset,t_offset], [-0.1, 1.1], "k:", alpha=0.5)

ax2.plot(time, m1, '--', lw = 2, c = color_m1_MF)
ax2.plot(time, m2, '--',  lw = 2, c = color_m2_MF)

ax2.set_xlim([0,t_max])
ax2.set_ylim([-0.1,1.1])
ax2.set_xlabel("time [s]")
#ax2.set_ylabel("$\hat{C}=%s$"%C_hat,labelpad=90, rotation=0)
ax2.set_ylabel("Overlaps")
"""
ax2.axes.get_xaxis().set_visible(False)



C_hat=0.3
data = np.loadtxt("../Output_full_sim_distr_gains/overlap_time_neus%s_gamma%s_P%s_tau%s_dt%s_chat%s_A%s_b%s.dat"%(N, gamma, P, tau, h, C_hat, A, b), unpack=True)
#time, m1, m2, stab = np.loadtxt("../../AttractorNetwork/output_2pop/2pop_with_noise_vs_time_gamma%s_time_step%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_Chat%s.dat"%(gamma, h, resolution, resolution_factor, rm, A, h0, b, load, C_hat), unpack = True, skiprows=1, delimiter = ",")
for i in range(1,P+1):
    ax3.plot(data[0,:],data[i,:], lw=linewidth)
ax3.plot([t_onset,t_onset], [-0.1, 1.1], "k:", alpha=0.5)
ax3.plot([t_offset,t_offset], [-0.1, 1.1], "k:", alpha=0.5)

#ax3.plot(time , m1 , '--', lw = 2, c = color_m1_MF, label = 'MF1')
#ax3.plot(time , m2 , '--',  lw = 2, c = color_m2_MF, label = 'MF2')

ax3.set_xlim([0,t_max])
ax3.set_ylim([-0.1,1.1])

ax3.set_xlabel("time [s]")
#ax3.set_ylabel("$\hat{C}=%s$"%C_hat,labelpad=90, rotation=0)
ax3.set_ylabel("Overlaps")
#plt.legend()

plt.savefig("../Figures_full_sim_distr_gains/N%s_gamma%s_P%s_A%s_beta%s_h0%s_tau%s_h%s_ton%s_toff%s_C_hat%s.pdf"%(N, gamma, P, A, beta, h0, tau, h, t_onset, t_offset, C_hat),   transparent=True)
plt.show()




fig_width = 5
fig_height =5
fig_size =  [fig_width,fig_height]
params = {'backend': 'TkAgg',
    'axes.labelsize': 40,
    'text.fontsize': 30,
    'axes.titlesize':40,
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

plt.figure()
mu = 0 # mean of distribution
sigma = 0.0141230
noise_distr = np.loadtxt("../Output_full_sim_distr_gains/final_noise_ditr_N%s_gamma%s_P%s_tau%s_dt%s_chat%s_A%s_b%s.dat"%(N, gamma, P, tau, h, C_hat, A, b), unpack=True)
print( len(noise_distr))

num_bins = 100
n, bins, patches = plt.hist(noise_distr, num_bins, facecolor='blue', alpha=0.5) #,  normed=1)
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.show()
