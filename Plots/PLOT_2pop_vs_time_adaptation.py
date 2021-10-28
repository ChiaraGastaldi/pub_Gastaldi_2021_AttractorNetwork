import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
import time
from matplotlib import gridspec

# these are the measures for fig 9
fig_width = 5
fig_height = 4
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

"""
# these are the measures for fig 10
fig_width = 12
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
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
"""

# ------ parameters ------------------------------------
# Model params
rm = 1.
A =  100.
b = 100  #0.1   #0.82
h0 = 0.25
gamma = 0.002
h = 0.1
min_J0 = 0.7
max_J0 = 1.5
C_hat = 0.04
load = 0.
# Simulation param
bound_low = -0.2
bound_up = 1.2
resolution = 500
t_max = 100
resolution_factor = 1   #playes with the precision of the nullclines
# (1=full precision, less then one= less precise)
#resolution in the x-variable (C_hat in this case)

data = np.loadtxt("../Output_2pop_adapt/MF_adapt_vs_time_p2_gamma%s_time_step%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s.dat"%(gamma, h, resolution, resolution_factor, rm, A, h0, b,  load), unpack=True, skiprows=1, delimiter = ",")

t, J0, mean_phi, phi11, phi10, phi01, phi00 = np.loadtxt("../Output_2pop_adapt/check_J0.dat", unpack=True)
t, thr1, thr2 = np.loadtxt("../Output_2pop_adapt/check_thr.dat", unpack=True)


plt.figure()
gs = gridspec.GridSpec(1,1, width_ratios=[1],height_ratios=[1],left=0.15, bottom=0.3, right=0.95, top=0.90, wspace=0.3, hspace=0.5)
plt.close('all')
ax0 = plt.subplot(gs[0])

MS = 1
Alpha = 1
colors = ['r', 'g', 'b']
#c = data[3,:]
ax0.plot(data[0,:], data[1,:] ,lw = 2, label = 'm1')
#cols = np.unique(data[3,:])
#print cols
ax0.plot(data[0,:], data[2,:] , lw = 2,label = 'm2')
ax0. plot(t, J0, 'k--', alpha = 0.5, label = 'J0')
#ax0. plot(t, np.divide(J0,gamma) * mean_phi, label = 'Inhibition')
#ax0. plot(t,  mean_phi, label = 'Mean activity')
#ax0. plot(t,  phi11, label = r'$\phi_{11}$')
#ax0. plot(t,  phi10,'-.',linewidth = 2, label = r'$\phi_{10}$')
#ax0. plot(t,  phi01,':',linewidth = 2, label = r'$\phi_{01}$')
#ax0. plot(t,  phi00,'--',linewidth = 2 , label = r'$\phi_{00}$')
#ax0. plot(t, thr1, '--',linewidth = 3, label = r'$thr_{11}$')
#ax0. plot(t, thr2, label = r'$thr_{10}$')
#ax0.grid()
ax0.set_ylim([bound_low, bound_up])
ax0.set_xlim([0, t_max])

ax0.set_ylabel(r'Overlaps')
ax0.set_xlabel(r'Time [AU]')


#leg = plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
# "violet = unstable, green = saddle, yellow = stable"
#ax0.set_title("yellow = stable")

plt.savefig( "../Figures_2pop_adapt/bifurcation_2pop_vs_t_time_step%s_resolution_pp_%s_resolution_factor_%s_rm%s_A%s_h0%s_b%s_gamma%s_load%s_minJ0%s_maxJ0%s_Chat%s.pdf" %(h, resolution, resolution_factor, rm, A, h0, b, gamma, load, min_J0, max_J0, C_hat),   transparent=True)
#plt.show()



