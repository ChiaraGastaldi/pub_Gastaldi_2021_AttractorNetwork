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

gamma = 0.002
N = 20000
P = 5
A = 1.
rm = 1.
beta = 100.
h0 = 0.25
tau = 1
h = 0.1
t_max = 15

load = P/N
resolution = 200
resolution_factor = 1
b = beta

linewidth= 2
color_m1_MF =   '#194A8D'
color_m2_MF =  '#AF601A'

t_vect=np.arange(0.,t_max,h)


plt.figure()
#gs = gridspec.GridSpec(4,1, width_ratios=[1],height_ratios=[1,1,1,1],left=0.23, bottom=0.12, right=0.97, top=0.9, wspace=0.1, hspace=0.1)
#plt.close('all')
#ax0 = plt.subplot(gs[0])
#ax1 = plt.subplot(gs[1])
#ax2 = plt.subplot(gs[2])
#ax3 = plt.subplot(gs[3])

#### graph 0 ####

C_hat=0.1

plt.title("Overlaps' dynamics")
data = np.loadtxt("../Output_full_sim/Exp_proposed_overlap_time_neus%s_gamma%s_P%s_tau%s_dt%s_chat%s_A%s_rm%s_b%s.dat"%(N, gamma, P, tau, h, C_hat, A, rm, b), unpack=True)
#time, m1, m2, stab = np.loadtxt("../../AttractorNetwork/output_2pop/2pop_with_noise_vs_time_gamma%s_time_step%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_Chat%s.dat"%(gamma, h, resolution, resolution_factor, rm, A, h0, b, load, C_hat), unpack = True, skiprows=1, delimiter = ",")
#time, m1, m2, rp, stab = np.loadtxt("../Output_2pop/MF_vs_time_gamma%s_time_step%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_Chat%s.dat"%(gamma, h, resolution, resolution_factor, rm, A, h0, b, load, C_hat), unpack = True, skiprows=1, delimiter = ",")
for i in range(2,3+2):
    plt.plot(data[0,:],data[i,:], lw = linewidth, label = 'Person %s'%(i-1))
for i in range(3+2,5+2):
    plt.plot(data[0,:],data[i,:], lw = linewidth, label = 'Contex %s'%(i-4))
#ax0.plot(data[0,:],data[1,:], 'k--', lw=linewidth, alpha = 0.5, label = r'$\sigma$ FS')
print( "final sigma", data[1,-1])

plt.plot([1 ,1], [-0.1, 1.1], "k:", alpha=0.5)
plt.plot([2,2], [-0.1, 1.1], "k:", alpha=0.5)
plt.plot([4,4], [-0.1, 1.1], "k:", alpha=0.5)
plt.plot([5,5], [-0.1, 1.1], "k:", alpha=0.5)
#plt.plot([11+h,11+h ], [-0.1, 1.1], "k:", alpha=0.5)


#ax0.plot(time, m1, '--', lw = 2, c = color_m1_MF)
#ax0.plot(time, m2, '--',  lw = 2, c = color_m2_MF)

plt.xlim([0,t_max])
plt.ylim([-0.1,1.1])
#ax0.plot([0, t_max], [0.0141230,0.0141230], 'b:', label = r'$\sigma$ MF')
#ax0.set_ylabel("$\hat{C}=%s$"%C_hat,labelpad=90, rotation=0, x=-1)
plt.ylabel("Overlaps")
#plt.axes.get_xaxis().set_visible(False)


plt.xlim([0,t_max])
plt.ylim([-0.1,1.1])

plt.xlabel("time [s]")
#plt.set_ylabel("$\hat{C}=%s$"%C_hat,labelpad=90, rotation=0)
plt.ylabel("Overlaps")
plt.legend()

plt.savefig("../Figures_full_sim/Predicted_experiement_with weak_stim0.03_toP2P3_N%s_gamma%s_P%s_A%s_rm%s_beta%s_h0%s_tau%s_h%s_C_hat%s.pdf"%(N, gamma, P, A, rm, beta, h0, tau, h, C_hat),   transparent=True)
plt.show()


