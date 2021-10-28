import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
import time
from matplotlib import gridspec

fig_width = 12
fig_height = 3
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
# ------ parameters ------------------------------------
# Model params
rm = 1
A =  1
b = 300  #0.1   #0.82
h0 = 0
gamma = 0.002
h = 0.1
min_J0 = 0.7
max_J0 = 1.2
C_hat = 0.04
load = 0
# Simulation param
bound_low = -0.1
bound_up = 1.1
resolution = 200
resolution_factor = 1   #playes with the precision of the nullclines
# (1=full precision, less then one= less precise)
#resolution in the x-variable (C_hat in this case)
t_max = 500

data = np.loadtxt("../Output_4pop_adapt/MF_adapt_vs_time_p4_gamma%s_time_step%s_resolution%s_factor%s_rm%s_A%s_h0%s_b%s_load%s.dat"%(gamma, h, resolution, resolution_factor, rm, A, h0, b,  load), unpack=True, skiprows=1, delimiter = ",")

t, J0, mean_phi, rec_mean_phi, phi1111, phi0000, phi1110, phi1101, phi1011, phi0111, phi1100, phi1010, phi0101, phi1001, phi0110, phi0011, phi1000, phi0100, phi0010, phi0001 = np.loadtxt("../Output_4pop_adapt/J0_vs_time.dat", unpack=True)
t, thr1, thr2 = np.loadtxt("../Output_4pop_adapt/thresholds_vs_time.dat", unpack=True)


plt.figure()
gs = gridspec.GridSpec(1,1, width_ratios=[1],height_ratios=[1],left=0.15, bottom=0.3, right=0.95, top=0.90, wspace=0.3, hspace=0.5)
plt.close('all')
ax0 = plt.subplot(gs[0])

MS = 1
Alpha = 1
colors = ['r', 'g', 'b']
#c = data[3,:]
ax0.plot(data[0,:], data[1,:], marker = ',', label = 'm1')
#cols = np.unique(data[3,:])
#print cols
ax0.plot(data[0,:], data[2,:] ,  marker = ',', label = 'm2')
ax0.plot(data[0,:], data[3,:] , marker = ',', label = 'm3')
ax0.plot(data[0,:], data[4,:], marker = ',', label = 'm4')
#ax0. plot(t, J0,  'k--', label = 'J0', alpha = 0.2)
#ax0. plot(t, np.divide(J0,gamma) * mean_phi, label = 'Inhibition')
#ax0. plot(t,  J0 * mean_phi/ gamma, label = 'Mean activity')
#ax0. plot(t,  J0 * rec_mean_phi/ gamma,":", label = 'Rec Mean activity')
#ax0. plot(t,  phi1111,'--',linewidth = 2 , label = r'$\phi_{1111}$')
#ax0. plot(t,  phi0000, label = r'$\phi_{0000}$')
#ax0. plot(t,  phi1110, ':', label = r'$\phi_{1110}$')
#ax0. plot(t,  phi1101, ':', label = r'$\phi_{1101}$')
#ax0. plot(t,  phi1011, ':', label = r'$\phi_{1011}$')
#ax0. plot(t,  phi0111, ':', label = r'$\phi_{0111}$')
##ax0. plot(t,  phi1100, ':', label = r'$\phi_{1100}$')  #!!!!
#ax0. plot(t,  phi1010, ':', label = r'$\phi_{1010}$')
#ax0. plot(t,  phi0101, ':', label = r'$\phi_{0101}$')
#ax0. plot(t,  phi1001, ':', label = r'$\phi_{1001}$')
#ax0. plot(t,  phi0110, ':', label = r'$\phi_{0110}$')
#ax0. plot(t,  phi0011, ':', label = r'$\phi_{0011}$')
#ax0. plot(t,  phi1000, ':', label = r'$\phi_{1000}$')
#ax0. plot(t,  phi0100, ':', label = r'$\phi_{0100}$')
#ax0. plot(t,  phi0010, ':', label = r'$\phi_{0010}$')  ### !!!!
#ax0. plot(t,  phi0001,':',linewidth = 2, label = r'$\phi_{0001}$')
#ax0. plot(t, thr1, '--',linewidth = 3, label = r'$thr_{11}$')
#ax0. plot(t, thr2, label = r'$thr_{10}$')
#ax0.grid()

ax0.set_ylabel(r'Overlap')
ax0.set_xlabel(r'Time')
ax0.set_ylim([-0.1, 1.1])
ax0.set_xlim([0, t_max])


leg = plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
# "violet = unstable, green = saddle, yellow = stable"
#ax0.set_title("yellow = stable")

plt.savefig( "../Figures_4pop_adapt/bifurcation_4pop_vs_t_time_step%s_resolution_pp_%s_resolution_factor_%s_rm%s_A%s_h0%s_b%s_gamma%s_load%s_minJ0%s_maxJ0%s_Chat%s.pdf" %(h, resolution, resolution_factor, rm, A, h0, b, gamma, load, min_J0, max_J0, C_hat),   transparent=True)
#plt.show()



