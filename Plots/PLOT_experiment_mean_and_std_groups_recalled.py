import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
import time
from matplotlib import gridspec
import itertools
import random
import scipy.io
import time

fig_width = 10
fig_height = 7
fig_size =  [fig_width,fig_height]

params = {'backend': 'TkAgg',
    'axes.labelsize': 20,
    #'text.fontsize': 20,
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

N = 10000
c_hat = 0.04
gamma = 0.002
N_trial = 2
max_group_size = 45

data = np.loadtxt("../Files/parent_how_many_neurons_respond_to_how_many_concepts_N%s_trials%s.dat"%(N, N_trial), unpack = True)
print( np.shape(data))
data = data.T
data_rnd = np.loadtxt("../Files/random_how_many_neurons_respond_to_how_many_concepts_N%s_trials%s.dat"%(N, N_trial), unpack = True)
data_rnd = data_rnd.T
data_ind = np.loadtxt("../Files/ind_how_many_neurons_respond_to_how_many_concepts_N%s_trials%s.dat"%(N, N_trial), unpack = True)
data_ind = data_ind.T
data_strict = np.loadtxt("../Files/strict_how_many_neurons_respond_to_how_many_concepts_N%s_trials%s.dat"%(N, N_trial), unpack = True)
data_strict = data_strict.T
#print data_ind
print( np.shape(data))
mean = np.mean(data, axis=1 )
norm = np.sum(mean)
std = np.std(data, axis=1 )
mean_rnd = np.mean(data_rnd, axis=1 )
norm_rnd = np.sum(mean_rnd)
std_rnd = np.std(data_rnd, axis=1 )
mean_ind = np.mean(data_ind, axis=1 )
norm_ind = np.sum(mean_ind)
std_ind = np.std(data_ind, axis=1 )
mean_strict = np.mean(data_strict, axis=1 )
norm_strict = np.sum(mean_strict)
std_strict = np.std(data_strict, axis=1 )
# only SU
'''
full_data_vector = [8.04268293e-01, 9.63414634e-02, 3.59756098e-02, 2.31707317e-02,
                    9.75609756e-03, 6.09756098e-03, 7.31707317e-03, 4.87804878e-03,
                    1.21951220e-03, 1.82926829e-03, 6.09756098e-04 , 6.09756098e-04,
                    1.21951220e-03 ,1.21951220e-03, 2.43902439e-03, 0.00000000e+00,
                    1.21951220e-03, 6.09756098e-04 ,0.00000000e+00, 0.00000000e+00,
                    6.09756098e-04, 0.00000000e+00, 6.09756098e-04, 0.00000000e+00,
                    0.00000000e+00 ,0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                    0.00000000e+00, 0.00000000e+00]
TOT_NEUS = 1640
prefactor_data = float(TOT_NEUS) /(TOT_NEUS - TOT_NEUS*full_data_vector[0])
number_concepts_neus_respond_to_parent = [9.82208323e-01, 1.39257441e-02, 5.01479632e-04, 3.59188430e-04,
                      3.30057540e-04, 4.16020105e-04, 3.15019150e-04, 2.55673586e-04,
                      1.55947233e-04, 1.22806552e-04, 1.29819102e-04, 9.91405288e-05,
                      1.39914923e-04, 9.09271998e-05, 1.70498107e-04, 9.66704721e-05,
                      7.44798452e-05, 5.94683182e-05, 8.77119249e-05, 7.01328623e-05,
                      3.90078700e-05, 5.65608582e-05, 1.36197329e-05, 4.03120308e-05,
                      1.04450406e-05, 2.05939618e-05, 5.09213350e-05, 4.48238355e-05,
                      1.18660264e-05, 5.14365424e-06]
number_concepts_neus_respond_to_random = [8.64281685e-01, 1.14075712e-01, 1.79222762e-02, 3.08978574e-03,
                      5.29705939e-04, 8.58602781e-05, 1.29222439e-05, 1.79395106e-06,
                      2.29043279e-07, 2.68409495e-08, 2.88169994e-09, 2.82941705e-10,
                      2.53684355e-11, 2.07488425e-12, 1.54729562e-13, 1.05194673e-14,
                      6.52143373e-16, 3.68783389e-17, 1.90297971e-18, 8.96286614e-20,
                      3.85344555e-21, 1.51208876e-22, 5.41304249e-24, 1.76647132e-25,
                      5.24895943e-27, 1.41793702e-28, 3.47507981e-30, 7.70665380e-32,
                      1.54153664e-33, 2.77011558e-35]
'''
# SU + MU
full_data_vector = [8.30791933e-01, 8.38662076e-02, 3.07427447e-02, 1.67240531e-02,
     8.85391048e-03, 6.88637482e-03, 5.41072307e-03, 4.18101328e-03,
     1.47565175e-03, 1.96753566e-03, 4.91883915e-04, 1.22970979e-03,
     1.22970979e-03, 9.83767831e-04, 9.83767831e-04, 7.37825873e-04,
     7.37825873e-04, 7.37825873e-04, 2.45941958e-04, 2.45941958e-04,
     2.45941958e-04, 0.00000000e+00, 4.91883915e-04, 0.00000000e+00,
     0.00000000e+00, 4.91883915e-04, 0.00000000e+00, 0.00000000e+00,
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    2.45941958e-04]
TOT_NEUS = 4066.0
prefactor_data = float(TOT_NEUS) /(TOT_NEUS - TOT_NEUS*full_data_vector[0])
number_concepts_neus_respond_to_parent =  [9.82258952e-01, 1.38806241e-02, 4.94996811e-04, 3.72246390e-04,
 3.22241815e-04, 4.32939559e-04, 3.13667399e-04, 1.93977672e-04,
 1.31567884e-04, 1.32315678e-04, 1.78396120e-04, 1.31782539e-04,
 1.39500665e-04, 7.45428161e-05, 1.59777347e-04, 1.03221851e-04,
 8.82252927e-05, 5.94262732e-05, 9.41478625e-05, 5.80891214e-05,
 3.59816676e-05, 3.54574204e-05, 1.62628854e-05, 5.93776485e-05,
 1.21854662e-05, 2.24307588e-05, 4.18472303e-05, 3.98993220e-05,
 8.17321094e-06, 5.41793439e-06, 2.47687386e-05, 5.29043872e-05,
 7.68426189e-16, 7.03560727e-14, 5.21472389e-12, 3.00620153e-10,
 1.26465172e-08, 3.45354225e-07, 4.59549505e-06, 5.81206905e-08,
 1.47096778e-06]
strict_parent =  [7.76673812e-01, 1.75794791e-01, 3.79993351e-02, 7.76486775e-03,
 1.46660827e-03, 2.53762014e-04, 4.01441680e-05, 5.80703933e-06,
 7.68209354e-07, 9.28832749e-08, 1.02517299e-08, 1.03147639e-09,
 9.45024694e-11, 7.87911574e-12, 5.97727186e-13, 4.12690195e-14,
 2.59443887e-15, 1.48593088e-16, 7.75729251e-18, 3.69267224e-19,
 1.60311370e-20, 6.34661828e-22, 2.29036490e-23, 7.52890742e-25,
 2.25184551e-26, 6.11861938e-28, 1.50729026e-29, 3.35777922e-31,
 6.74260855e-33, 1.21565300e-34, 1.95836252e-36, 2.80217146e-38,
 3.53529245e-40, 3.89681525e-42, 3.70953521e-44, 3.00441660e-46,
 2.02956353e-48, 1.11255203e-50, 4.75418120e-53, 1.48564885e-55,
 3.01961148e-58]
number_concepts_neus_respond_to_random =  [8.65193796e-01, 1.13311572e-01, 1.78064915e-02, 3.06280947e-03,
 5.24852065e-04, 8.54219402e-05, 1.29661661e-05, 1.82204471e-06,
 2.36122338e-07, 2.81361803e-08, 3.07414095e-09 ,3.07180704e-10,
 2.80148230e-11, 2.32875256e-12, 1.76324892e-13 ,1.21591644e-14,
 7.63815039e-16, 4.37253635e-17, 2.28198943e-18 ,1.08607998e-19,
 4.71447294e-21, 1.86628261e-22, 6.73468211e-24 ,2.21374980e-25,
 6.62099809e-27, 1.79899188e-28 ,4.43164830e-30 ,9.87221050e-32,
 1.98237163e-33 ,3.57406489e-35, 5.75761078e-37 ,8.23836401e-39,
 1.03936781e-40, 1.14564983e-42, 1.09058670e-44 ,8.83282994e-47,
 5.96680312e-49, 3.27083772e-51, 1.39770075e-53 ,4.36771781e-56,
 8.87747523e-59]


x1 = np.linspace(1, 41, 41)
x = np.linspace(1, 45, 45)
y = np.linspace(1, len(full_data_vector)-1, len(full_data_vector)-1)
plt.figure()
# theory prediction
plt.plot(x1,number_concepts_neus_respond_to_parent, 's--', color = 'deepskyblue', ms = 10, label= 'Indicator predicted')
plt.plot(x1,strict_parent, 'o--', color = 'limegreen', ms = 10, label= 'Hierarchical predicted')


rescaled_data = prefactor_data*np.array(full_data_vector[1:len(full_data_vector)])


#plt.errorbar(x, mean/norm, yerr = std/norm, fmt ='s', c = 'royalblue', alpha = 0.8, ms = 5, label= 'Indicator neuron model')
#plt.errorbar(x, mean_rnd/norm_rnd, yerr = std_rnd/norm_rnd, fmt ='^',  ms = 5, c = 'red', alpha = 0.8, label= 'Compact correlation model')
#plt.errorbar(x, mean_ind/norm_ind, yerr = std_ind/norm_ind, fmt ='>', c = 'black', alpha = 0.8, label= 'indicators')
#plt.errorbar(x, mean_strict/norm_strict, yerr = std_strict/norm_strict, fmt ='o', ms = 5, c = 'green', alpha = 0.8, label= 'Hierarchical generative model')

plt.plot(x, mean/norm,  'bs-', ms = 10, label = 'Indicator neuron model')
plt.fill_between(x, mean/norm + std/norm, mean/norm - std/norm, color = 'b', alpha = 0.3)
plt.plot(x, mean_rnd/norm_rnd,  'r^-', ms = 10, label = 'Compact correlation model')
plt.fill_between(x, mean_rnd/norm_rnd + std_rnd/norm_rnd, mean_rnd/norm_rnd - std_rnd/norm_rnd, color = 'r', alpha = 0.3)
plt.plot(x, mean_strict/norm_strict,  'go-', ms = 10, label = 'Hierarchical generative model')
plt.fill_between(x, mean_strict/norm_strict + mean_strict/norm_strict, mean_strict/norm_strict- mean_strict/norm_strict, color = 'g', alpha = 0.3)
plt.plot(y,rescaled_data , 'k*', ms = 10, label = 'data')

plt.xlabel('number of concepts a neu responds to')
plt.ylabel('probability')
plt.yscale('log')
plt.grid()
plt.legend()
plt.ylim([10**(-5), 2])
plt.savefig("../Figures_exp/mean_and_std_groups_recalled_N%s_trials%s.pdf"%(N, N_trial),   transparent=True)
plt.show()





plt.close('all')
plt.figure()
fin_x = np.linspace(1, 10, 10)
plot_data = [0]*10
for i in range(10):
    plot_data[i] = rescaled_data[i]
for i in range(10, len(rescaled_data)):
    plot_data[9] += rescaled_data[i]
    
plot_mean_parent = [0]*10
plot_std_parent = [0]*10
for i in range(10):
    plot_mean_parent[i] = (mean/norm)[i]
    plot_std_parent[i] = (std/norm)[i]
for i in range(10, len(mean)):
    plot_mean_parent[9] += (mean/norm)[i]
    plot_std_parent[9] += ((std/norm)[i])**2
plot_std_parent[9] = (plot_std_parent[9])**(1./2.)   #(len(mean)-10))
plot_err_parent  = (np.array(plot_mean_parent) + np.array(plot_std_parent))

plot_mean_rnd = [0]*10
plot_std_rnd = [0]*10
for i in range(10):
    plot_mean_rnd[i] = (mean_rnd/norm_rnd)[i]
    plot_std_rnd[i] = (std_rnd/norm_rnd)[i]
for i in range(10, len(mean_rnd)):
    plot_mean_rnd[9] += (mean_rnd/norm_rnd)[i]
    plot_std_rnd[9] += ((std_rnd/norm_rnd)[i])**2
plot_std_rnd[9] = (plot_std_rnd[9])**(1./2.)   #(len(mean)-10))
plot_err_rnd  = (np.array(plot_mean_rnd) + np.array(plot_std_rnd))

plot_mean_strict = [0]*10
plot_std_strict = [0]*10
for i in range(10):
    plot_mean_strict[i] = (mean_strict/norm_strict)[i]
    plot_std_strict[i] = (std_strict/norm_strict)[i]
for i in range(10, len(mean_strict)):
    plot_mean_strict[9] += (mean_strict/norm_strict)[i]
    plot_std_strict[9] += ((std_strict/norm_strict)[i])**2
plot_std_strict[9] = (plot_std_strict[9])**(1./2.)   #(len(mean)-10))
plot_err_strict  = (np.array(plot_mean_strict) + np.array(plot_std_strict))
    
plt.plot(fin_x, plot_data,  'k*-', ms = 10, label = ' ') #c')
'''
plt.plot(fin_x, plot_mean_parent,  'bs-', ms = 10, label = 'parent')
print len(fin_x), len(plot_mean_parent), len(plot_err_parent)
plt.fill_between(fin_x, plot_mean_parent, plot_err_parent, color = 'b', alpha = 0.5)
'''
plt.errorbar(fin_x, plot_mean_parent, yerr = plot_std_parent , lolims=True , fmt ='-s', ms = 10, c = 'blue', alpha = 1, label= ' ')# Indicator neuron model' )
'''
plt.plot(fin_x, plot_mean_rnd,  'r^-', ms = 10, label = 'random')
plt.fill_between(fin_x, plot_mean_rnd, plot_err_rnd, color = 'r', alpha = 0.5)
'''
plt.errorbar(fin_x, plot_mean_rnd, yerr = plot_std_rnd , lolims=True , fmt ='-^', ms = 10, c = 'red', alpha = 1, label= ' ') #Compact correlation model' )
'''
plt.plot(fin_x, plot_mean_strict,  'go-', ms = 10, label = 'strict')
plt.fill_between(fin_x, plot_mean_strict, plot_err_strict, color = 'g', alpha = 0.5)
'''
plt.errorbar(fin_x, plot_mean_strict, yerr = plot_std_strict , lolims=True , fmt ='-o', ms = 10, c = 'green', alpha = 1, label= ' ') #Hierarchical generative model' )

plt.grid()
plt.legend()
plt.ylim([10**(-5), 2])
plt.xlabel('Number of concepts a neu responds to')
plt.ylabel('Probability')
plt.yscale('log')
plt.xticks(np.arange(min(fin_x), max(fin_x)+1, 1.0))
plt.savefig('../Figures_exp/Final_fig_N%s_trials%s.pdf'%(N,N_trial),   transparent=True)
plt.show()
