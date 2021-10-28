import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
import time
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

# ------ parameters ------------------------------------
rm= 1 #76.2
A= 1 #3.55 #3.55 #1. # 3.55
b= 1000 #0.82 #0.1   #0.82

gamma = 0.01 #0.00025
h0= 0.25
load = 0.
C_hat = 0

# Simulation param
bound_low = - gamma
bound_up = gamma
resolution = 200
resolution_factor=1  #playes with the precision of the nullclines (1=full precision, less then one= less precise)
# ------------------------------------------

lightblue='#1976D2' #'#0099FF'
orange=  '#E67E22'  #'#FF9100'
green='#66CC00'     #'#009900'


print("Start loading the data")
#xp,  zp = np.loadtxt("output1/fp3D_1pop_noise%s%s%s%s%s%s%s%s.dat"%(gamma, size, resolution, resolution_factor, rm, A, h0, C_hat), unpack=True)
M11, M21 = np.loadtxt("../Output_1pop/nullclineM_1pop_noise_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_load%s.dat"%(gamma,  resolution, resolution_factor, rm, A, h0,load), unpack=True)
R11, R21 = np.loadtxt("../Output_1pop/nullclineR_1pop_noise_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_load%s.dat"%(gamma,  resolution, resolution_factor, rm, A, h0, load), unpack=True)
xfp, yfp = np.loadtxt("../Output_1pop/fp3D_1pop_noise_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_load%s.dat"%(gamma, resolution, resolution_factor, rm, A, h0,load), unpack=True)
stability = np.loadtxt("../Output_1pop/Stab_1pop_noise_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_load%s.dat"%(gamma, resolution, resolution_factor, rm, A, h0, load), unpack=True)

print("Finished loading the data")
def sigmoid(x):
    return 1./(1.+np.exp(-b*(x*rm-h0)))
               
x_lin = np.linspace(-0.1,1.1, 1000)

fig_width = 10
fig_height =7
fig_size =  [fig_width,fig_height]
params = {'backend': 'TkAgg',
    'axes.labelsize': 30,
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

plt.figure()

Colors=["b", "g", "r", "c"]
stab_legend = ["Stable FP", "Saddle FP", "Unstable FP"]

plt.plot(R11, R21, '.', color=orange, ms=3  ,alpha=1, label=r'$r$-nullcline')
plt.plot(M11, M21, '.', color=lightblue, ms=3 ,alpha=1, label=r'$m^1$-nullcline')

for j in range(len(xfp)):
    plt.plot(xfp,yfp, ".", color=Colors[int(stability[j]+1)], label  = "%s"%(stab_legend[int(stability[j]+1)]))
plt.title(r'load, $\alpha=%s$'%load)
plt.ylabel(r'R')
plt.xlabel(r'$m^1$')
plt.grid()
#plt.legend()
plt.savefig("../Figures_1pop/PP_m1R_b%s_load%s_gamma%s_rm%s_h0%s_resolution%s.png"%(b,load,gamma,rm,h0, resolution ),   transparent=True)
#plt.show()
