import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
import time
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig_width = 10
fig_height = 10
fig_size =  [fig_width,fig_height]
params = {'backend': 'TkAgg',
    'axes.labelsize': 30,
    'axes.titlesize':30,
    'legend.fontsize': 30,
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
load = 0
A = 1
rm = 1
resolution = 100
factor = 1.1
size = 50
extent = [0.1 , 1.2, 0 , 200]

#t_vect=np.arange(0,t_max-h,h)

C_crit = np.loadtxt("../Output_2pop/critical_corr_vs_b_h0_gamma%s_size%s_resolution%s_factor%s_rm%s_A%s_load%s.dat"%( gamma, size,  resolution, factor, rm, A, load), unpack=True)

plt.close()
plt.figure()

c_crit = gamma + (1.-gamma)*C_crit
##cmap = 'inferno'
image = plt.imshow(c_crit, origin = 'lower' , aspect = 0.005, extent = extent,  cmap = 'hot', alpha = 0.9, vmin= 0 , vmax=1)#, interpolation = 'gaussian') #0.00167
plt.colorbar(image, label=r'$c^*$',fraction=0.042, pad=0.04)
plt.xlabel(r"$h_0$")
plt.ylabel("b")
"""
#plt.title("C_hat_crit")
#plt.xticks([0.01,1.])
#plt.yticks([40.,1000.])
#plt.xticklabels(['0.01','1'])
"""
CS = plt.contour( C_crit, extent = extent, cmap = 'hot') #

plt.clabel(CS, CS.levels[0:7], inline=True, fontsize= 20, colors='k' )#, manual = True, fmt= r"$\hat{C}^*$ = %1.3f", orientation = 'horizontal')
##plt.grid()
plt.savefig("../Figures_2pop/Crit_C_hat_vs_b_h0_gamma%s_A%s_rm%s.pdf"%(gamma, A, rm))
plt.show()


"""
plt.figure()
ax = plt.gca()
im = ax.imshow(C_crit, origin = 'lower' , aspect = 0.001, extent = extent) #, interpolation = 'gaussian')
ax.set_xlabel(r"$h_0$")
ax.set_ylabel("b")
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
CS = ax.contour( C_crit, extent = extent)
ax.clabel(CS, inline=1, fontsize= 19, colors='k' )
plt.colorbar(im, cax=cax)
"""

fig_width = 5
fig_height = 3
fig_size =  [fig_width,fig_height]
params = {'backend': 'TkAgg',
    'axes.labelsize': 30,
    'axes.titlesize':30,
    'legend.fontsize': 30,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'text.usetex': False,
    'font.family': 'sans-serif',
    'figure.figsize': fig_size,
    'font.size': 30}
rcParams.update(params)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
h0 = np.linspace(0.1, 1.2, 100)
plt.figure()
C_hat_extracted_b100 = C_crit[25, :]
c_extracted_b100 = gamma + (1.-gamma)*C_hat_extracted_b100
plt.plot(h0, c_extracted_b100, lw = 2, c = 'k')
plt.yticks([0, 0.5, 1])
plt.xticks([0, 0.5, 1])
plt.ylim([-0.01, 1.01])
plt.xlim([-0.01, 1.2])
plt.ylabel(r"$c^*$")
plt.xlabel(r"$h_0$")
plt.savefig("../Figures_2pop/Crit_C_hat_vs_h0_gamma%s_A%s_rm%s_b%s.pdf"%(gamma, A, rm, 100),   transparent=True)
plt.show()
