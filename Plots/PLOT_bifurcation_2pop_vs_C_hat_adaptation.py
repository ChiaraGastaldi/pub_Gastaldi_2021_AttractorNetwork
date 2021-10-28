import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
import time
from matplotlib import gridspec
from itertools import groupby
import matplotlib.patches as mpatches

fig_width = 11
fig_height = 6
fig_size =  [fig_width,fig_height]
params = {'backend': 'TkAgg',
    'axes.labelsize': 40,
    'text.fontsize': 30,
    'axes.titlesize':30,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'text.usetex': False,
    'font.family': 'sans-serif',
    'figure.figsize': fig_size,
    'font.size': 30}
#rcParams.update(params)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ------ parameters ------------------------------------
# Model params
rm = 1
A =  1
b = 30  #0.1   #0.82
h0 = 0.
J0 = 0.7
gamma = 0.002

# Simulation param
bound_low = -0.05
bound_up = 1.05
#c_hat = 0.64
resolution = 300
resolution_factor = 1   #playes with the precision of the nullclines
# (1=full precision, less then one= less precise)
size = 50             #resolution in the x-variable (C_hat in this case)
load = 0

def extract_critical_C_hat(data_cells):
    C_hat = data_cells[0, :]
    x_p = data_cells[1, :]
    r_p = data_cells[2, :]
    stab = data_cells[3, :]
    C_hat_crit = 0.
    for i in range(len(x_p)):
        if stab[i] == 1.0 and x_p[i] > 0.15: # for the min C_hat
            C_hat_crit = C_hat[i]
            return C_hat_crit
        #if stab[i] == 0.0:    # for the max C_hat
        #   C_hat_crit = C_hat[i]
    return C_hat_crit

data = np.loadtxt("../Output_2pop_adapt/Bifurcation_plot_vs_C_gamma%s_size%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_J0%s.dat"%(gamma, size, resolution, resolution_factor, rm, A, h0, b,  load, J0), unpack=True, skiprows=1, delimiter = " ")
c_hat = extract_critical_C_hat(data)
print( "../Output_2pop_adapt/Bifurcation_plot_vs_C_gamma%s_size%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_J0%s.dat"%(gamma, size, resolution, resolution_factor, rm, A, h0, b,  load, J0))

plt.figure()
gs = gridspec.GridSpec(1,1, width_ratios=[1], height_ratios = [1], left=0.15, bottom=0.2, right=0.6, top=0.95, wspace=0.2, hspace=0.02)
plt.close('all')
ax0 = plt.subplot(gs[0])

MS = 3.
Alpha = 1.
colors = ['r', '#7fb800', '#0d2c54']
#ax0.scatter(data[0,:], data[1,:],  c = data[3,:], marker = '.')#, ms = MS, alpha = Alpha, label='Saddle point')  #data[3,:]
#cols = np.unique(data[3,:])
#print cols
c = gamma + (1 - gamma) * data[0,:]
for i in range(len(data[0,:])):  #c[i]
    ax0.plot(c[i], data[1,i],  c = colors[int(data[3,i])],  alpha = Alpha, ms = MS, marker = '.')


ax0.set_ylabel(r'$m^1$')
ax0.set_xlabel(r'$c$')
ax0.set_ylim([-0.2, 1.2])
ax0.set_xlim([0, 1])

ax0.plot([c_hat, c_hat], [-1.5, 1.5], 'k--', label=r'$\hat{C}^*\sim$%s' %c_hat)

#red_patch = mpatches.Patch(color='red', label='The red data')
#plt.legend(handles=[red_patch])

leg = plt.legend( bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., frameon=False, fontsize = 30)
# "violet = unstable, green = saddle, yellow = stable"
#ax0.set_title("yellow = stable")
plt.grid()
plt.savefig( "../Figures_2pop_adapt/bifurcation_2pop_vs_c_hat_size%s_resolution_pp_%s_resolution_factor_%s_rm%s_A%s_h0%s_b%s_gamma%s_load%s_J0%s.pdf" %(size, resolution, resolution_factor, rm, A, h0, b, gamma, load, J0),   transparent=True)
plt.show()



