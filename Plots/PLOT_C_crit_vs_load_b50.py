import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
from matplotlib import gridspec

fig_width = 14.
fig_height = 8.
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
rm = 1 #76.2 #0.83
A =  1  #3.55
b = 50  #0.82  #4.355  #0.1   #0.82
h0 = 0.25  #2.46
gamma = 0.002
size = 50
resolution = 100
# ------------- COLORS -------------
color1 = "#323031"
color2 = "#db3a34"
color3 = "#177e89"

alphas=[0 , 0.1  , 0.25, 0.5  , 0.75, 1 ]
chats1_Naive=[ 0.157142857143, 0.15306122449, 0.142857142857, 0.132653061224, 0.122448979592, 0.117346938776 ]         # h0 = 0.25
chats2_Naive=[0.408163265306, 0.408163265306, 0.397959183673, 0.387755102041, 0.377551020408, 0.367346938776 ]               # h0 = 0.5
chats1_TAP=[ 0.15306122449, 0.15306122449, 0.142857142857, 0.132653061224, 0.127551020408, 0.117346938776 ]           # h0 = 0.25
chats2_TAP=[0.408163265306, 0.408163265306, 0.397959183673, 0.387755102041, 0.387755102041, 0.377551020408 ]                 # h0 = 0.5
chats1_Naive_corr=[ 0.157142857143, 0.15306122449, 0.142857142857, 0.132653061224, 0.122448979592, 0.117346938776 ]   # h0 = 0.25
chats2_Naive_corr=[0.408163265306, 0.408163265306, 0.397959183673, 0.387755102041, 0.377551020408, 0.367346938776 ]        # h0 = 0.5

plt.close('all')
plt.figure()
gs = gridspec.GridSpec(2,2, width_ratios=[1, 1],height_ratios=[1,1],left=0.1, bottom=0.2, right=0.65, top=0.8, wspace=0.3, hspace=0.5)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
ax0.plot(alphas, chats1_Naive , c = color1, lw=1, alpha = 0.5)
ax0.plot(alphas, chats1_Naive , c = color1, marker = '*', ms = 12)
ax0.plot(alphas, chats1_TAP , '--', c = color2, lw=1, alpha = 0.5)
ax0.plot(alphas, chats1_TAP , '--', c = color2, marker = 'o', ms = 7)
ax0.plot(alphas, chats1_Naive_corr , ':', c = color3, lw=1, alpha = 0.5)
ax0.plot(alphas, chats1_Naive_corr , ':', c = color3, marker = '<', ms = 7)
ax0.set_xlabel(r"$\alpha$")
ax0.set_ylabel(r"$\hat{C}^*$")
ax0.set_ylim([0,0.25])
ax0.set_xlim([-0.05,1.05])
ax0.set_title(r"$h_0 = 0.25$")
ax1.plot(alphas, chats2_Naive , c = color1 , lw=1,  alpha = 0.5)
ax1.plot(alphas, chats2_Naive , c = color1 , marker = '*', ms = 12, label = "Mean field")
ax1.plot(alphas, chats2_TAP , '--', c = color2, lw=1, alpha = 0.5)
ax1.plot(alphas, chats2_TAP , '--', c = color2, marker = 'o', ms = 7, label = "No self-interaction")
ax1.plot(alphas, chats2_Naive_corr , ':', c = color3, lw=1,  alpha = 0.5)
ax1.plot(alphas, chats2_Naive_corr , ':', c = color3, marker = '<', ms = 7, label = "Correlated noise")
ax1.set_xlabel(r"$\alpha$")
#ax1.set_ylabel(r"$\hat{C}^*$")
ax1.set_ylim([0,0.5])
ax1.set_xlim([-0.05,1.05])
#ax1.get_yaxis().set_visible(False)
ax1.set_title(r"$h_0 = 0.5$")
ax1.legend(bbox_to_anchor = (1.05, 1.), loc = 2, borderaxespad = 0., frameon=False)


# --------------- SECOND PART ----------------
pop = [2 , 3 ,4 ]

# resolution = 500, resolution_factor = 1.1,   size = 50
#chats1 =[ 0.157142857143, 0.158163265306 ]        # h0 = 0.25
#chats2 =[0.408163265306, 0.418367346939  ]         # h0 = 0.5\

# resolution = 100, resolution_factor = 1.1,   size = 50
#correlation :
#chats1 =[ 0.157142857143,0.163265306122, 0.168367346939]        # h0 = 0.25
# number of shared nerons
chats1 = [0.158828571429, 0.16493877551, 0.170030612245 ]
# correlation
#chats2 =[ 0.415306122449, 0.418367346939 ,  0.428571428571]         # h0 = 0.5
# number of shared neurons
chats2 = [0.416475510204 , 0.419530612245 ,0.429714285714]

ax2.plot(pop, chats1 , c = color1 , lw=1, alpha = 0.5)
ax2.plot(pop, chats1 , c = color1 , marker = '*', ms = 12)
ax2.set_xlabel("p")
ax2.set_ylabel(r"$\hat{C}^*$")
ax2.set_ylim([0,0.25])
ax2.set_xlim([1.9, 4.1])
ax3.set_xticks([2, 3, 4])

ax3.plot(pop, chats2 , c = color1 , lw=1, alpha = 0.5)
ax3.plot(pop, chats2 , c = color1 , marker = '*', ms = 12)
ax3.set_xlabel("p")
#ax3.set_ylabel(r"$\hat{C}^*$")
ax3.set_ylim([0,0.5])
ax3.set_xlim([1.9, 4.1])
ax3.set_xticks([2, 3, 4])
#ax3.get_yaxis().set_visible(False)



plt.savefig("C_hat_vs_alpha.pdf")
plt.show()

