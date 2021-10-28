import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
import time
from matplotlib import gridspec

fig_width = 10
fig_height = 6
fig_size =  [fig_width,fig_height]
params = {'backend': 'TkAgg',
    'axes.labelsize': 40,
    'axes.titlesize':30,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'text.usetex': False,
    'font.family': 'sans-serif',
    'figure.figsize': fig_size,
    'font.size': 30}
rcParams.update(params)

# ------ parameters ------------------------------------
# Model params
rm = 1
A =  1
b = 100  #0.1   #0.82
h0 = 0.25
gamma = 0.002

# Simulation param
bound_low = -0.05
bound_up = 1.05
resolution = 100
resolution_factor = 1   #playes with the precision of the nullclines
# (1=full precision, less then one= less precise)
size = 100              #resolution in the x-variable (C_hat in this case)
c_hat = 0

data = np.loadtxt("../Output_1pop/bifurcation_1pop_fixed_points_vs_load_gamma%s_size%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_chat%s.dat"%(gamma, size, resolution, resolution_factor, rm, A, h0, b,  c_hat), unpack=True,  delimiter = ",")

plt.figure()
gs = gridspec.GridSpec(1,1, width_ratios=[1], height_ratios = [1], left=0.15, bottom=0.2, right=0.65, top=0.95, wspace=0.2, hspace=0.02)
plt.close('all')
ax0 = plt.subplot(gs[0])

MS = 3
Alpha = 1
colors = ['r', 'g', 'b']
for i in range(size):
    ax0.plot(data[0,i], data[1,i],  c = colors[int(data[3,i]-1)], marker = '.', ms = MS, alpha = Alpha, label='Saddle point')

ax0.set_ylabel(r'$m^1$')
ax0.set_xlabel(r'$\alpha$')
ax0.set_ylim([-0.05, 1.05])

#alpha = 0.41
#ax0.plot([alpha, alpha], [-1.5, 1.5], 'k--', label=r'$\alpha^*\sim$%s' %alpha)

#leg = plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
ax0.grid()

plt.savefig( "../Figures_1pop/bifurcation_1pop_vs_load_size%s_resolution_pp_%s_resolution_factor_%s_rm%s_A%s_h0%s_b%s_gamma%s_chat%s.png" %(size, resolution, resolution_factor, rm, A, h0, b, gamma, c_hat))
#plt.show()



