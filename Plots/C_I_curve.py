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
size = 10
resolution  = 150
factor = 1.1
rm = 1
A = 1
h0 = 0.25
b = 100
load = 0

fin_C = []
fin_I = []
#t_vect=np.arange(0,t_max-h,h)
C, I = np.loadtxt("../Output_2pop/generate_I_C_curve_gamma%s_rm%s_A%s_h0%s_b%s_load%s.dat"%(gamma, rm, A, h0, b, load), unpack=True)

#for i in range(len(C)):
#    if

plt.figure()
gs = gridspec.GridSpec(1,1, width_ratios=[1],height_ratios=[1],left=0.15, bottom=0.3, right=0.95, top=0.90, wspace=0.3, hspace=0.5)
plt.close('all')
ax0 = plt.subplot(gs[0])
ax0.plot(C,I)

ax0.set_xlim([0, 0.25])
#ax0.set_ylim([0, 0.25])


plt.savefig("../Figures_2pop/C_I_curve.pdf",   transparent=True)
#plt.show()
