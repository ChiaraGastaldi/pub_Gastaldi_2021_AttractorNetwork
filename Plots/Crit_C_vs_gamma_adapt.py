import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
import time
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

fig_width = 6
fig_height = 6
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
#rcParams.update(params)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

size = 50
resolution  = 150
factor = 1.1
rm = 1.
A = 1.
h0 = 0.
b = 100.
load = 0
minJ0 = 0.7
maxJ0 = 1.2

#t_vect=np.arange(0,t_max-h,h)
gamma_vect,  C_min_crit,   C_max_crit = np.loadtxt("../Output_2pop_adapt/Cs_vs_gamma_size%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_MinJ0%s_MaxJ0%s.dat"%(size, resolution, factor, rm, A, h0, b, load, minJ0, maxJ0 ), unpack=True)

#print gamma_vect
plt.figure()
gs = gridspec.GridSpec(1,1, width_ratios=[1],height_ratios=[1],left=0.15, bottom=0.3, right=0.95, top=0.90, wspace=0.3, hspace=0.5)
plt.close('all')
ax0 = plt.subplot(gs[0])
x1 = []
y1 = []
x2 = []
y2 = []
x1 += [gamma_vect[0]]
y1 += [C_min_crit[0]]
x2 += [gamma_vect[0]]
y2 += [C_max_crit[0]]
for i in range(1, len(gamma_vect)):
    if C_min_crit[i] != C_min_crit[i-1]:
        #print C_min_crit[i], C_min_crit[i-1]
        x1 += [gamma_vect[i]]
        y1 += [C_min_crit[i]]
    if C_max_crit[i] != C_max_crit[i-1]:
        x2 += [gamma_vect[i]]
        y2 += [C_max_crit[i]]
    
x1 += [gamma_vect[-1]]
y1 += [C_min_crit[-1]]
x2 += [gamma_vect[-1]]
y2 += [C_max_crit[-1]]
#ax0.plot(gamma_vect,  C_max_crit , c = 'mediumpurple', label = r'Max $c^*$', alpha = 0.5)
#ax0.plot(gamma_vect,  C_min_crit , c = 'darkturquoise', label = r'Min $c^*$', alpha = 0.5)
ax0.plot(x2,  y2 , c = 'mediumpurple', label = r'Max $c^*$')
ax0.plot(x1,  y1 , c = 'darkturquoise', label = r'Min $c^*$')
ax0.set_xlabel(r'$\gamma$')
ax0.set_ylabel(r'$c^*$')
ax0.set_ylim([-0.1, 1.1])
ax0.set_xlim([0.001, 0.01])
plt.legend()
plt.grid()

plt.savefig("../Figures_2pop_adapt/Bifurcation_plot_vs_C_size%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_MinJ0%s_MaxJ0%s.pdf"%(size, resolution, factor, rm, A, h0, b, load, minJ0, maxJ0 ),   transparent=True)
plt.show()



plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

size = 50
resolution  = 150
factor = 1.1
rm = 1.
A = 1.
h0 = 0.
b = 100.
load = 0
minJ0 = 0.7
maxJ0 = 1.2

#t_vect=np.arange(0,t_max-h,h)
# it is the actual small c
gamma_vect = [0.001, 0.002, 0.005, 0.01, 0.1]
c_min_crit = [ 0.06 ,0.06, 0.06, 0.06, 0]
c_max_crit = [ 0.66 ,0.665 , 0.665 , 0.68 , 0.87]

plt.close('all')
#print gamma_vect
plt.figure()
gs = gridspec.GridSpec(1,1, width_ratios=[1],height_ratios=[1],left=0.15, bottom=0.3, right=0.95, top=0.90, wspace=0.3, hspace=0.5)
ax0 = plt.subplot(gs[0])

ax0.plot(gamma_vect,  c_max_crit , 's-', c = 'mediumpurple', label = r'Max $c^*$')
ax0.plot(gamma_vect,  c_min_crit , '^:', c = 'darkturquoise', label = r'Min $c^*$')

ax0.set_xlabel(r'$\gamma$')
ax0.set_ylabel(r'$c^*$')
ax0.set_ylim([-0.1, 1.1])
ax0.set_xlim([0.0008, 0.11])
ax0.set_xscale('log')
plt.legend()
#plt.grid()

plt.savefig("../Figures_2pop_adapt/HomeMadeBifurcation_plot_vs_C_size%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_MinJ0%s_MaxJ0%s.pdf"%(size, resolution, factor, rm, A, h0, b, load, minJ0, maxJ0 ),   transparent=True)
plt.show()
