import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
import time
from matplotlib import gridspec

fig_width = 12
fig_height = 5
fig_size =  [fig_width,fig_height]
params = {'backend': 'TkAgg',
    'axes.labelsize': 20,
    'text.fontsize': 20,
    'axes.titlesize':20,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'text.usetex': False,
    'font.family': 'sans-serif',
    'figure.figsize': fig_size,
    'font.size': 20}
rcParams.update(params)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

#start_time = time.time()

# ------ parameters ------------------------------------
rm = 1. #76.2 #0.83
A =  1.  #3.55
b = 100.  #0.82  #4.355  #0.1   #0.82
h0 = 0.25  #2.46
gamma = 0.002
C_hat = 0.1
load = 0.
h = 0.1
t_onset = 0.5
t_offset = 8
t_max = 0.2


# Simulation param
bound_low = -0.1
bound_up = 1.1
resolution = 1000
resolution_factor = 1   #playes with the precision of the nullclines
# (1=full precision, less then one= less precise)



lightblue = '#00a6ed' #'#0099FF'
orange = '#FF8C00'  #FF9100'
green =  '#66CC00'     #'#009900'
MS = 10


fig = plt.figure(0)
gs = gridspec.GridSpec(nrows = 1, ncols = 3, width_ratios=[1,1,1],height_ratios=[1],left=0.1, bottom=0.2, right=0.95, top=0.85, wspace=0.05, hspace=0.05)
plt.close('all')
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])


# First PP
I1 = 0.
M11, M21 = np.loadtxt("../Output_2pop/nullcline1_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_C_hat%s_I1%s.dat"%(gamma, resolution, resolution_factor, rm, A, h0, b, load, C_hat, I1), unpack = True)
M12, M22 = np.loadtxt("../Output_2pop/nullcline2_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_C_hat%s_I1%s.dat"%(gamma, resolution, resolution_factor, rm, A, h0, b, load, C_hat, I1), unpack = True)
xp, yp, R, stab = np.loadtxt("../Output_2pop/fp_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_C_hat%s_I1%s.dat"%(gamma, resolution, resolution_factor, rm, A, h0, b, load, C_hat, I1), unpack=True)
x, y, vx, vy = np.loadtxt("../Output_2pop/Quiver_PP_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_C_hat%s_I1%s.dat"%(gamma, resolution, resolution_factor, rm, A, h0, b, load, C_hat, I1), unpack = True)
time, m1, m2, r, stab = np.loadtxt("../Output_2pop/MF_vs_time_gamma%s_time_step%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_Chat%s.dat"%(gamma, h, resolution, resolution_factor, rm, A, h0, b, load, C_hat), unpack = True, skiprows=1, delimiter = ",")

#vx = vx / np.sqrt(vx**2 + vy**2);
#vy = vy / np.sqrt(vx**2 + vy**2);
ax0.quiver(x, y, vx, vy , alpha = 0.6)
ax0.plot(M11, M21, '.', color= orange, ms=2  ,alpha=1, label=r'$m^1$-nullcline')  #'#ff8c00'
ax0.plot(M12, M22, '.', color= lightblue, ms=2  , alpha=1, label=r'$m^2$-nullcline') #'#0091ea'
ax0.plot(xp,yp, 'k.')
for t in range(len(time)):
    if time[t] < t_onset:
        ax0.plot(m1[t], m2[t], 'g.', ms = MS)
ax0.set_title('Before stim.')
ax0.set_ylabel(r'$m^1$')
ax0.set_xlabel(r'$m^2$')
ax0.set_ylim([bound_low-0.05,bound_up+0.05])
ax0.set_xlim([bound_low-0.05,bound_up+0.05])

# Second PP
I1 = 1.
M11, M21 = np.loadtxt("../Output_2pop/nullcline1_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_C_hat%s_I1%s.dat"%(gamma, resolution, resolution_factor, rm, A, h0, b, load, C_hat, I1), unpack = True)
M12, M22 = np.loadtxt("../Output_2pop/nullcline2_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_C_hat%s_I1%s.dat"%(gamma, resolution, resolution_factor, rm, A, h0, b, load, C_hat, I1), unpack = True)
xp, yp, R, stab = np.loadtxt("../Output_2pop/fp_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_C_hat%s_I1%s.dat"%(gamma, resolution, resolution_factor, rm, A, h0, b, load, C_hat, I1), unpack=True)
x, y, vx, vy = np.loadtxt("../Output_2pop/Quiver_PP_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_C_hat%s_I1%s.dat"%(gamma, resolution, resolution_factor, rm, A, h0, b, load, C_hat, I1), unpack = True)
time, m1, m2, R, stab = np.loadtxt("../Output_2pop/MF_vs_time_gamma%s_time_step%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_Chat%s.dat"%(gamma, h, resolution, resolution_factor, rm, A, h0, b, load, C_hat), unpack = True, skiprows=1, delimiter = ",")

#vx = vx / np.sqrt(vx**2 + vy**2);
#vy = vy / np.sqrt(vx**2 + vy**2);
ax1.quiver(x, y, vx, vy , alpha = 0.6)
ax1.plot(M11, M21, '.', color= orange, ms=2  ,alpha=1, label=r'$m^1$-nullcline')  #'#ff8c00'
ax1.plot(M12, M22, '.', color= lightblue, ms=2  , alpha=1, label=r'$m^2$-nullcline') #'#0091ea'
ax1.plot(xp,yp, 'k.')
for t in range(len(time)):
    if time[t] > t_onset and time[t] < t_offset:
        ax1.plot(m1[t], m2[t], 'g.', ms = MS)
ax1.set_title('During stim.')
ax1.set_ylabel(r'$m^1$')
ax1.set_xlabel(r'$m^2$')
ax1.set_ylim([bound_low-0.05,bound_up+0.05])
ax1.set_xlim([bound_low-0.05,bound_up+0.05])
ax1.get_yaxis().set_visible(False)

# Third PP
I1 = 0.
M11, M21 = np.loadtxt("../Output_2pop/nullcline1_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_C_hat%s_I1%s.dat"%(gamma, resolution, resolution_factor, rm, A, h0, b, load, C_hat, I1), unpack = True)
M12, M22 = np.loadtxt("../Output_2pop/nullcline2_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_C_hat%s_I1%s.dat"%(gamma, resolution, resolution_factor, rm, A, h0, b, load, C_hat, I1), unpack = True)
xp,yp, R, stab = np.loadtxt("../Output_2pop/fp_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_C_hat%s_I1%s.dat"%(gamma, resolution, resolution_factor, rm, A, h0, b, load, C_hat, I1), unpack=True)
x, y, vx, vy = np.loadtxt("../Output_2pop/Quiver_PP_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_C_hat%s_I1%s.dat"%(gamma, resolution, resolution_factor, rm, A, h0, b, load, C_hat, I1), unpack = True)
time, m1, m2, R, stab = np.loadtxt("../Output_2pop/MF_vs_time_gamma%s_time_step%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_Chat%s.dat"%(gamma, h, resolution, resolution_factor, rm, A, h0, b, load, C_hat), unpack = True, skiprows=1, delimiter = ",")

#vx = vx / np.sqrt(vx**2 + vy**2);
#vy = vy / np.sqrt(vx**2 + vy**2);
ax2.quiver(x, y, vx, vy , alpha = 0.6)
ax2.plot(M11, M21, '.', color= orange, ms=2  ,alpha=1, label=r'$m^1$-nullcline')  #'#ff8c00'
ax2.plot(M12, M22, '.', color= lightblue, ms=2  , alpha=1, label=r'$m^2$-nullcline') #'#0091ea'
ax2.plot(xp,yp, 'k.')
for t in range(len(time)):
    if  time[t] > t_offset:
        ax2.plot(m1[t], m2[t], 'g.', ms = MS)
ax2.set_title('After stim.')
ax2.set_ylabel(r'$m^1$')
ax2.set_xlabel(r'$m^2$')
ax2.set_ylim([bound_low-0.05,bound_up+0.05])
ax2.set_xlim([bound_low-0.05,bound_up+0.05])
ax2.get_yaxis().set_visible(False)



# final
leg=plt.legend(bbox_to_anchor = (0.05, -1), loc = 2, borderaxespad = 0., frameon=False, fontsize = 30)
#plt.suptitle(r"$J_0 = %s$ " %J0)
#plt.colorbar(image)
plt.savefig("../Figures_2pop/PP_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_Chat%s.pdf"%(gamma, resolution, resolution_factor, rm, A, h0,b, load,  C_hat),   transparent=True)
plt.show()



fig_width = 15
fig_height = 4
fig_size =  [fig_width,fig_height]
params = {'backend': 'TkAgg',
    'axes.labelsize': 20,
    'text.fontsize': 20,
    'axes.titlesize':20,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'text.usetex': False,
    'font.family': 'sans-serif',
    'figure.figsize': fig_size,
    'font.size': 20}
rcParams.update(params)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

time, m1, m2, R, stab = np.loadtxt("../Output_2pop/MF_vs_time_gamma%s_time_step%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_Chat%s.dat"%(gamma, h, resolution, resolution_factor, rm, A, h0, b, load, C_hat), unpack = True, skiprows=1, delimiter = ",")

plt.figure()
I1 = 0.
plt.plot(time, m1, lw = 2, label = r"$m^1$")
plt.plot(time, m2, lw = 2, label = r"$m^2$")
plt.plot(time, r, 'k--', lw = 2, alpha = 0.5, label = r"$m^2$")
#idx = next(x[0] for x in enumerate(m2) if x[1] > h0)
#print "time gap is ", (time[idx] - t_onset)
plt.grid()

print r[-1],"   ", r[0]
plt.xlabel("Time [s]")
plt.ylabel("Overlaps")
plt.ylim([-0.1, 1.1])
plt.xlim([0, t_max])
leg = plt.legend( bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., frameon=False, fontsize = 20)
plt.savefig("../Figures_2pop/MF_vs_time_gamma%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s_Chat%s.pdf"%(gamma, resolution, resolution_factor, rm, A, h0,b, load,  C_hat),   transparent=True)
plt.show()

