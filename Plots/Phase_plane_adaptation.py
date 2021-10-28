import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
import time
from matplotlib import gridspec

fig_width = 12
fig_height =8
fig_size =  [fig_width,fig_height]
params = {'backend': 'TkAgg',
    'axes.labelsize': 40,
    #'text.fontsize': 30,
    'axes.titlesize':40,
    'legend.fontsize': 30,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'text.usetex': False,
    'font.family': 'sans-serif',
    'figure.figsize': fig_size,
    'font.size': 30}
rcParams.update(params)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

#start_time = time.time()

# ------ parameters ------------------------------------
rm = 1. #76.2 #0.83
A =  10. #3.55
b = 100 #4.35  #0.1   #0.82
h0 = 0.25  #1.7  #2.46
gamma = 0.002
J0 = 0.

# Simulation param
bound_low = -0.2
bound_up = 1.2
resolution = 500

resolution_factor = 1   #playes with the precision of the nullclines
# (1=full precision, less then one= less precise)
load = 0.
C_hat = 0.




#lightblue =  '#1976D2' #'#0099FF' #
lightblue = '#00a6ed'
#orange = '#E67E22'  #'#FF9100'
orange =  '#f6511d'
#green = '#66CC00'     #'#009900'
green = '#00ff15'

fig = plt.figure(0)
gs = gridspec.GridSpec(1,1, width_ratios=[1],height_ratios=[1],left=0.15, bottom=0.15, right=0.6, top=0.85, wspace=0.3, hspace=0.05)
plt.close('all')
ax0 = plt.subplot(gs[0])


M11, M21 = np.loadtxt("../Output_2pop_adapt/nullcline1_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_C_hat%s.dat"%(gamma, resolution, resolution_factor, rm, A, h0, b, load, C_hat), unpack = True)
xp,yp, stab = np.loadtxt("../Output_2pop_adapt/fp.dat", unpack=True)
x, y, vx, vy = np.loadtxt("../Output_2pop_adapt/quiver.dat", unpack = True)

#vx = vx / np.sqrt(vx**2 + vy**2);
#vy = vy / np.sqrt(vx**2 + vy**2);

ax0.plot(M11, M21, '.', color= 'darkorange' , ms= 7  ,alpha=1, label=r'$m^1$-nullcline')  #'#ff8c00'
ax0.plot(M21, M11, '.', color= lightblue, ms= 7  , alpha=1, label=r'$m^2$-nullcline') #'#0091ea'
colors = ['r', '#7fb800', '#0d2c54']
for i in range(len(xp)):
    ax0.plot(xp[i], yp[i],  c = colors[int(stab[i])], alpha = 1, ms = 10, marker = 'o')
ax0.quiver(x, y, vx, vy , alpha = 0.6)


ax0.set_title(r'$\hat{C}=%s$'%C_hat)
ax0.set_ylabel(r'$m^1$')
ax0.set_xlabel(r'$m^2$')
ax0.set_ylim([bound_low,bound_up])
ax0.set_xlim([bound_low,bound_up])

# final
leg=plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., frameon=False, fontsize = 30)
#plt.suptitle(r"$J_0 = %s$ " %J0)
#plt.colorbar(image)


plt.savefig("../Figures_2pop_adapt/PP_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_J0%s_Chat%s.pdf"%(gamma, resolution, resolution_factor, rm, A, h0,b, load, J0, C_hat),   transparent=True)

plt.show()
