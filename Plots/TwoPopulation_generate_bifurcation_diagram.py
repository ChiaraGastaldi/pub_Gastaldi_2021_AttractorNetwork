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

# ------ parameters ------------------------------------
# Model params
rm = 1 # 0.83  #76.2
A =  1 #3.55
b = 12.8 #4.35   #0.82  #0.1   #0.82
h0 = 0.57  #1.7  #2.46
gamma = 0.002

# Simulation param
bound_low = -0.05
bound_up = 1.05
resolution = 150

resolution_factor = 1  #playes with the precision of the nullclines
# (1=full precision, less then one= less precise)
size = 100              #resolution in the x-variable (C_hat in this case)
load = 0.
corr_noise = "false"
model = "AttractorNetwork.TwoPopulations.MeanField()"
# NOTE: In alternative you can use the following model, if you did include the corrections to exclude self interaction
#model = "AttractorNetwork.TwoPopulations.MFnoSelfInteraction()"

# ------------------- functions to extract the critical correlation ------------------------------------
def extract_critical_C_hat(data_cells):
    C_hat = data_cells[0, :]
    x_p = data_cells[1, :]
    r_p = data_cells[2, :]
    stab = data_cells[3, :]
    C_hat_crit = 0.
    for i in range(len(x_p)):
        if x_p[i] > 0.9    and stab[i] == 1.0:
        #if x_p[i] > 0.9  and (x_p[i] == x_p[i-1])  and stab[i] == 1.0:
            C_hat_crit = C_hat[i]
    return C_hat_crit
data = np.loadtxt("../Output_2pop/Bifurcation_plot_vs_C_gamma%s_size%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_model%s_corr_noise%s.dat"%(gamma, size, resolution, resolution_factor, rm, A, h0, b,  load, model, corr_noise), unpack=True, skiprows=1, delimiter = " ")
c_hat = extract_critical_C_hat(data)
print( "Critical correlation = ", c_hat)
c_star = gamma + (1.-gamma) * c_hat
print( "c_star = ", c_star)
# ------------------- PLOT -----------------------------
plt.figure()
gs = gridspec.GridSpec(1,1, width_ratios=[1], height_ratios = [1], left=0.15, bottom=0.2, right=0.6, top=0.95, wspace=0.2, hspace=0.02)
plt.close('all')
ax0 = plt.subplot(gs[0])

MS = 3.
Alpha = 1
colors = ['r', '#7fb800', '#0d2c54']
#ax0.scatter(data[0,:], data[1,:],  c = data[3,:], marker = '.')#, ms = MS, alpha = Alpha, label='Saddle point')  #data[3,:]
##cols = np.unique(data[3,:])
##print cols
c = gamma + (1 - gamma) * data[0,:]
for i in range(len(data[0,:])):
    if int(data[3,i]) != 0:
        ax0.plot(c[i], data[1,i],  c = colors[int(data[3,i])],  alpha = Alpha, ms = MS, marker = '.')
c_unstable = []
unstable = []
for i in range(len(data[0,:])):
    if int(data[3,i]) == 0:
        #ax0.plot(c[i], data[1,i],  c = colors[int(data[3,i])],  alpha = Alpha, ms = MS, marker = '.')
        c_unstable  += [c[i]]
        unstable += [data[1,i]]
ax0.plot(c_unstable, unstable,  'r--',  alpha = Alpha, ms = MS)

ax0.set_ylabel(r'$m^1$')
ax0.set_xlabel(r'$c$')
ax0.set_ylim([-0.2, 1.2])
ax0.set_xlim([0., h0/rm + h0/(10*rm) ])

#ax0.plot([c_hat, c_hat], [-1.5, 1.5], 'k--', label=r'$\hat{C}^* = $%s' %c_hat)
ax0.plot([c_star, c_star], [-1.5, 1.5], 'k--', label=r'$c^* = $%s' %c_star)

#red_patch = mpatches.Patch(color='red', label='The red data')
#plt.legend(handles=[red_patch])
plt.yticks([0, 0.25, 0.5, 0.75, 1])
leg = plt.legend( bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., frameon=False, fontsize = 30)
# "violet = unstable, green = saddle, yellow = stable"
#ax0.set_title("yellow = stable")

plt.savefig( "../Figures_2pop/Bifurcation_vs_c_size%s_resolution_pp_%s_resolution_factor_%s_rm%s_A%s_h0%s_b%s_gamma%s_load%s_model%s.pdf" %(size, resolution, resolution_factor, rm, A, h0, b, gamma, load, model),   transparent=True)
plt.show()



