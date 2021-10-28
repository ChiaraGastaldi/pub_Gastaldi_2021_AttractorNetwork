import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import cm
import time
import math
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
import scipy.io
from scipy import signal
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import collections

"""
    README:
    File 'dataset_NatComm2016.mat'
    all data is in the structure 'data' which has subfields
    it contains the sub-fields:
    1) 'session'
    2) 'neurons_id': in each trial a N_neus x 2 matrix. The columns are the cluster and the chanenl number.
    3) 'responses': in each trial a N_neus x N_stim matrix of {0,1}, neurons x stimuli
    4) 'unit_classification': in each trial a N_neus x 1 matrix of arrays contaning a string, 'su' = single unit
    5) 'location': in each trial a  N_neus x 1 matrix with the acronim of the brain area. RAH = right anterior hippocampus, LAH = left anterior hippocampus, RAM = right amygdala, LAM = left amygdala, RPH = right posterior hippocampus, LPH = left posterior hippocampus.
    6) 'stimuli': in each trial a  N_stim x 4  matrix, containing the a) stim names, b) the jpg name, c) the cathegory, ex: 'FAMOUS_PEOPLE', d) an integer, the category code.
    7) 'web_association': in each trial, the  N_stim x N_stim  matrix of pairwise associations between the stimuli. Scores are centered on 0. Some entries are nan if the association score could not be extracted from the web scores, for example the associations between the patient's relatives.
    8) 'familiarity': in each trial, the  N_stim x N_stim  matrix 
    9) 'stimulus_name': in each trial, the  N_stim x 1 with the stimuli name, same as stimuli.a
    10) 'stimulus_category': in each trial, the  N_stim x 1 with the stimuli category, same as stimuli.c
    11) 'stimulus_category_code': in each trial, the  N_stim x 1 with the stimuli category code, same as stimuli.d
    12) 'neuron_classification': in each trial, the  N_neus x 1, with the neuron classification into 'unknown', 'pyramidal', 'interneuron'.
    13) 'Avg_Fr': : in each trial, the  N_neus x 1
    14) 'mean_spikeshape': in each trial, the  N_neus x 1
    All subfields have shape (1, 100) because there are 100 sessions, which can be considered independent.
    For example: In the first session we have N_neus = 7, N_stim = 94
"""

#mat = scipy.io.loadmat('Data/dataset_NatComm2016.mat')
mat = scipy.io.loadmat('Data/dataset_NatComm2016_with_MU.mat')
data = mat['data']
session = data['session']
neurons_id = data['neurons_id']
responses = data['responses']
unit_classification = data['unit_classification']
location = data['location']
stimuli = data['stimuli']
web_association = data['web_association']
familiarity = data['familiarity']
stimulus_name = data['stimulus_name']
stimulus_category = data['stimulus_category']
stimulus_category_code = data['stimulus_category_code']
neuron_classification = data['neuron_classification']
Avg_Fr = data['Avg_Fr']
mean_spikeshape = data['mean_spikeshape']

web = scipy.io.loadmat('Data/Web_association_matrix.mat')
#print web
hits_name = web['hits_name']
#print np.shape(hits_name)
print( 'shape of mat', np.shape(mat))
print( 'session = ', np.shape(session))
print( 'neurons_id = ',np.shape(neurons_id[0, 0]))
'''
#print neurons_id[0, 3]
print( 'responses = ',np.shape(responses[0, 0]))
#print responses[0, 0]
print( 'unit_classification = ',np.shape(unit_classification[0,0]))
#print unit_classification[0, 0]
print( 'location = ', np.shape(location[0,0]))
#print location[0,0]
print( 'stimuli = ', np.shape(stimuli[0,0]))
#print stimuli[0,0][1,0]
print( 'web_association = ',np.shape(web_association[0,0]))
#print float(web_association[0,0][0,0])
print( math.isnan(web_association[0,0][0,0] ))
print( 'familiarity = ', np.shape(familiarity[0,0]))
#print familiarity[0,0]
print( 'stimulus_name = ',np.shape(stimulus_name[0,0]))
#print stimulus_name[0,0] == stimulus_name[0,1]
print( 'stimulus_category = ', np.shape(stimulus_category[0,0]))
#print stimulus_category[0,0]
print( 'stimulus_category_code = ',np.shape(stimulus_category_code[0,0]))
#print stimulus_category_code[0,0]
print( 'neuron_classification = ',np.shape(neuron_classification[0,0]))
#print neuron_classification[0,0]
print( 'Avg_Fr = ', np.shape(Avg_Fr[0,0]))
#print Avg_Fr
print( 'mean_spikeshape = ',np.shape(mean_spikeshape[0,0]))
#print mean_spikeshape[0,0]
#print responses[0,1][0,:]
'''
#----------------------------------------------------------------------------------------------------
###################################### PARAMETERS ###################################################
#----------------------------------------------------------------------------------------------------
c = 0.04  #0.1950207468879668 #0.21241830065359477
# there is a dependence on c, for c = 0.2 the parent pattern distribution stabilizes at 10^-5 instead of 10^-6
gamma = 0.002  #0.3132530120481928 #0.3759398496240602  #0.002 #have to be bigger than c
C = (c - gamma)/(1. - gamma)
#----------------------------------------------------------------------------------------------------
########################################### Functions ###############################################
#----------------------------------------------------------------------------------------------------
def p1(a, b):
    return a * b + (1. - a) * (1. - b)
def p11(a, b):
    return a * b**2 + (1. - a) * (1. - b)**2
def p110(a, b):
    return a * b**2*(1. -b) + (1. - a) * b* (1. - b)**2
def p111(a, b):
    return a * b**3 + (1. - a) * (1. - b)**3
def p100(a, b):
    return a * b*(1. - b)**2 + (1. - a) * b**2 * (1. - b)

def get_alpha_beta(gamma, C_hat):
    beta_roots = np.roots([ 2.,  -3., (1 + 2 * gamma * (1 - gamma) * (1 - C_hat)), - gamma * (1 - gamma) * (1 - C_hat)])
    alpha_candidate = 0.
    beta_candidate = 0.
    for beta in beta_roots:
        if beta.imag == 0 and beta.real >= 0 and beta.real <= 1.000000001:
            beta = beta.real
            alpha = (gamma + beta - 1)/(2 * beta - 1)
            C_hat_recomputed=(p11(alpha, beta) - gamma**2)/(gamma*(1-gamma))
            gamma_recomputed = p1(alpha, beta)
            if alpha >= 0 and alpha <= 1.000000001 and abs(C_hat_recomputed - C_hat) < 0.001 and abs(gamma_recomputed - gamma) < 0.001:
                
                #if alpha_candidate != 0 :  # i.e. if \alpha_candidate is already assigned
                #    print("Using second values of alpha and beta. ")
                alpha_candidate = alpha
                beta_candidate = beta
    return alpha_candidate, beta_candidate

def prob_parent_pattern(p, n1): # p = number correlated patterns
    alpha, beta = get_alpha_beta(gamma, C)
    n0 = p - n1
    prefactor = float(math.factorial(p))/float((math.factorial(n1)*math.factorial(p-n1)))
    #print n1, n0, prefactor
    prob = prefactor*(alpha * beta**n1 * (1-beta)**n0 + (1. - alpha)*(1. - beta)**n1 * beta**n0)
    return prob
    
def prob_strict_parent_pattern(p, n1): # p = number correlated patterns
    alpha  = gamma/c
    n0 = p - n1
    prefactor = float(math.factorial(p))/float((math.factorial(n1)*math.factorial(p-n1)))
    #print n1, n0, prefactor
    if n1 == 0:
        prob = prefactor*(alpha * c**n1 * (1.-c)**n0 + (1. - alpha))
    else:
        prob = prefactor*(alpha * c**n1 * (1.-c)**n0)
    return prob

def prob_random_pattern(p ,n1): # p = number correlated patterns
    n0 = p - n1
    prefactor = float(math.factorial(p))/float((math.factorial(n1)*math.factorial(p-n1)))
    #print 'p ', p, 'n1 ', n1, 'p - n1', n0, 'prefactor', prefactor,' (c**n1 * (1. - c)**n0) ', (c**n1 * (1. - c)**n0)
    prob = 0
    if n1 == 0:
        prob = prefactor * ((1. - gamma) - gamma * (1. - c)**(p-1))
    else:
        prob = prefactor * gamma * c**(n1 -1)*(1. - c )**(p - (n1 - 1))
    return prob

def S_parent(k):
    return 1 - prob_parent_pattern(k, 0)
    
def S_strict_parent(k):
    return 1. - prob_strict_parent_pattern(k, 0)

def S_random(k):
    return 1 - prob_random_pattern(k, 0)

def hom_zeta(pmax, groups): # homogeneous zera vector
    sum = 0.
    #print 'hom_prob_groupsize(j, pmax)', hom_prob_groupsize(1, pmax)
    for j in range(1, pmax+1):
        sum += groups[j]
    #print 'sum', sum
    return  1./(1. + sum)

def Number_concepts_neus_respond_to(pmax, groups):  #this is the main differnece with the v1, we start from p = 1
    # pmax = max group size
    # group_sizes = vector containing the number of groups for each size (the dimention of the vector is pmax + 1)
    final_fit_parent = [0]*(pmax)
    final_fit_strict_parent = [0]*(pmax)
    final_fit_rnd = [0]*(pmax)
    
    # M is the renormalizaion constant for S
    sump = 0.
    sumr = 0.
    sums = 0.
    for j in range(1, pmax +1 ):
        #print(j)
        sump += (groups[j] * S_parent(j))
        sumr += (groups[j] * S_random(j))
        sums += (groups[j] * S_strict_parent(j))
    Mp = 1. / (sump)
    Mr = 1. / (sumr)
    Ms = 1. / (sums)

    for  p in range(1, pmax +1):
        for j in range(max(p,1),pmax +1):
            #print('p = ', p, ' and j = ', j)
            final_fit_parent[p-1] += Mp*groups[j] * prob_parent_pattern(j, p) * S_parent(j)
            final_fit_rnd[p-1] += Mr*groups[j] * prob_random_pattern(j, p) * S_random(j)
            final_fit_strict_parent[p-1] += Ms*groups[j] * prob_strict_parent_pattern(j, p) * S_strict_parent(j)
            #print('final_fit_parent_hom[1] = ', final_fit_parent_hom[1])
    factor_p  = np.sum(final_fit_parent)
    factor_r  = np.sum(final_fit_rnd)
    factor_s  = np.sum(final_fit_strict_parent)
    #print('this must be one ', 1./factor_p *np.sum(final_fit_parent_hom),1./factor_r*np.sum(final_fit_rnd_hom))
    return 1./factor_p *np.array(final_fit_parent), 1./factor_s *np.array(final_fit_strict_parent), 1./factor_r*np.array(final_fit_rnd)


# I made it strickly positive to avoid problems with the anticorrelations
def Distances_matrix(responses, session):
    #print('session = ', session)
    N_stim = 0
    if np.shape(responses[0,session])[0] > 0:
        N_stim = np.shape(responses[0,session][0,:])[0]
    else:
        N_stim = np.shape(responses[0,session][:])[1]
        #print('stimuli!!!!',np.shape(responses[0,session][:])[1])
    #print('N_stim',N_stim)
    Modified_associations = np.zeros([N_stim, N_stim])
    # here we create the positive association matrix without NaNs
    counts = 0
    web = []
    for stim1 in range(N_stim):
        for stim2 in range(N_stim):
            #print('----------- web score ',web_association[0,session][stim1, stim2])
            if math.isnan(web_association[0,session][stim1, stim2]) == True:
                # we set all NaNs to 0
                #print('it is nan')
                Modified_associations[stim1, stim2] = 0.
            else:
                #print('NOT nan')
                Modified_associations[stim1, stim2] = web_association[0,session][stim1, stim2]
                counts += 1
                web += [web_association[0,session][stim1, stim2]]
    print('average of the webassociations = ',np.sum(web)/counts, ' and sigma = ', np.var(web))
    if N_stim == 0:
        return N_stim, np.zeros([N_stim,N_stim]), 0.
    # the average is 0 and sigma is 1

    max_association = np.max(Modified_associations)
    min_association = np.min(Modified_associations)
    for stim1 in range(N_stim):
        for stim2 in range(N_stim):
            if stimulus_category[0,session][stim1] == 'UNKNOWN_PEOPLE' and stimulus_category[0,session][stim2] == 'UNKNOWN_PEOPLE':
                Modified_associations[stim1, stim2] = min_association
            if stimulus_category[0,session][stim1] == 'FAMILY' and stimulus_category[0,session][stim2] == 'FAMILY':
                Modified_associations[stim1, stim2] = 2.
            if stimulus_category[0,session][stim1] == 'UNKNOWN_PLACES' and stimulus_category[0,session][stim2] == 'UNKNOWN_PLACES':
                Modified_associations[stim1, stim2] = min_association
    # make it positive and normalized
    #print('Modified_associations =', Modified_associations)
    if (max_association - min_association) != 0.:
        Modified_associations = (Modified_associations - min_association)/(max_association - min_association)
    # make it a distance matrix instead of a similarity on
    Modified_associations = 1. - Modified_associations
    print('average of the final matrix = ',np.average(Modified_associations), ' and sigma = ', np.var(Modified_associations))
    return N_stim, Modified_associations, np.average(Modified_associations)



#----------------------------------------------------------------------------------------------------
##################  MAIN ############################################################################
#----------------------------------------------------------------------------------------------------
"""
full_data_vector = [8.04268293e-01, 9.63414634e-02, 3.59756098e-02, 2.31707317e-02,
                    9.75609756e-03, 6.09756098e-03, 7.31707317e-03, 4.87804878e-03,
                    1.21951220e-03, 1.82926829e-03, 6.09756098e-04 , 6.09756098e-04,
                    1.21951220e-03 ,1.21951220e-03, 2.43902439e-03, 0.00000000e+00,
                    1.21951220e-03, 6.09756098e-04 ,0.00000000e+00, 0.00000000e+00,
                    6.09756098e-04, 0.00000000e+00, 6.09756098e-04, 0.00000000e+00,
                    0.00000000e+00 ,0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                    0.00000000e+00, 0.00000000e+00]
TOT_NEUS = 1640
"""
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
'''
print(np.shape(responses[0,93]))
plt.figure()
plt.imshow(responses[0,93])
plt.colorbar()
plt.show()

print(stimulus_name[0,0]==stimulus_name[0,1])
for i in range(94):
    print(stimulus_name[0,0][i][0], stimulus_name[0,1][i][0])
'''
# ----------------- probs in one session -----------------------------------
"""
print("------------------------------------------------------------------------")
sess = 93 #93
#print(stimulus_category[0,sess])

N_stim, distance_matrix, mean = Distances_matrix(responses, sess)
print('N_stim  =', N_stim)
threshold = mean  #(1.+ mean)/2.  #
print('threshold = ', threshold)
# this function is for hierarchical clusering
model = 0.
if N_stim != 0:
    model = AgglomerativeClustering(affinity='precomputed', n_clusters=None, compute_full_tree = True, distance_threshold= threshold, linkage='complete').fit(distance_matrix)
    print(model.labels_)
    #dictionary of counts: Key = counted number, argument = number of counts
    counts = collections.Counter(model.labels_)
    #print("counts, ", counts)
    n_groups = len(counts)
    print('n_groups =', n_groups)
    stims_groups = []
    for i in range(n_groups):
        stims_groups += [counts[i]]
    #sorted_goups = sorted(stims_groups)  # no need to sort
    #print('stims_groups  ' ,stims_groups)
    pmax = np.max(stims_groups)
    groups = [0]*(pmax+1) # in entry the number of group which size is the entry: example groups[1] = # of groups of size 1
    for i in range(pmax+1):
        for j in stims_groups:
            if j==i:
                groups[i] += 1
    #print('groups =', groups)
    n_concepts_neus_respond_to_parent, n_concepts_neus_respond_to_random = Number_concepts_neus_respond_to(pmax, groups)

    x = np.linspace(1, len(n_concepts_neus_respond_to_parent), len(n_concepts_neus_respond_to_parent))
    print(x)
    y = np.linspace(1, len(full_data_vector)-1, len(full_data_vector)-1)
    plt.figure()
    #print(n_concepts_neus_respond_to_parent)
    plt.plot(x, n_concepts_neus_respond_to_parent, 'b^-', label= 'parent')
    plt.plot(x, n_concepts_neus_respond_to_random, 'r>--', label= 'random')
    print(np.sum(n_concepts_neus_respond_to_parent), np.sum(n_concepts_neus_respond_to_random))
    #plt.plot(full_data_vector, 'cs', label = 'all neus')
    # data
    prefactor = float(TOT_NEUS) /(TOT_NEUS - TOT_NEUS*full_data_vector[0])
    plt.plot(y, prefactor*np.array(full_data_vector[1:len(full_data_vector)]), 'g*', label = 'all neus')
    #print(np.sum(prefactor*np.array(full_data_vector[1:len(full_data_vector)])))
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.xlabel('number of concepts a neu responds to')
    plt.ylabel('probability')
    plt.savefig('Figures/phy_section93.pdf')
    plt.show()

"""
'''
#Check the distance_matrix
plt.figure()
plt.title('0 = equal')
plt.imshow(distance_matrix)
plt.colorbar()
plt.savefig('similarity_matrix93.pdf')
plt.show()
'''

# ----------------- check the final distribution = the other datafile -----------------------------------
'''
## check the final distribution - it works!
number_concepts_neus_respond_to = np.zeros(41)
number_concepts_neus_respond_to_interneus = np.zeros(41)
number_concepts_neus_respond_to_pyramidal = np.zeros(41)
for sess in range(100):
    N_neu = np.shape(responses[0,sess][:,0])[0]
    #print 'N_neu = ', N_neu
    number_of_resp =  np.zeros(N_neu)
    for neu in range(N_neu):
        number_of_resp[neu] = np.sum(responses[0,sess][neu,:])
        #print 'number_of_resp[neu] ', number_of_resp[neu]
        number_concepts_neus_respond_to[int(number_of_resp[neu])] += 1
        #print(neuron_classification[0,sess][:,0][0][0])
        if neuron_classification[0,sess][:,0][0][0] == 'interneuron':
            number_concepts_neus_respond_to_interneus[int(number_of_resp[neu])] += 1
        elif neuron_classification[0,sess][:,0][0][0] == 'pyramidal':
            number_concepts_neus_respond_to_pyramidal[int(number_of_resp[neu])] += 1
#print number_concepts_neus_respond_to
tot_neus = np.sum(number_concepts_neus_respond_to)
tot_neus_interneus = np.sum(number_concepts_neus_respond_to_interneus)
tot_neus_pyramidal = np.sum(number_concepts_neus_respond_to_pyramidal)
#mat = scipy.io.loadmat('Data/Resp_counts_SU.mat')
#print mat
#Nresp  = mat['Nresp']
#print Nresp
print(number_concepts_neus_respond_to/tot_neus)
print('tot neus = ', tot_neus)
plt.figure()
plt.plot(number_concepts_neus_respond_to/tot_neus, 's',label = 'all neus')
plt.plot(number_concepts_neus_respond_to_interneus/tot_neus_interneus,'^', label = 'interneus')
plt.plot(number_concepts_neus_respond_to_pyramidal/tot_neus_pyramidal, '*', label = 'pyramidal')
bins = np.linspace(-0.5,30.5,31)
#plt.hist(Nresp, bins = bins,label = 'Emanuela')
plt.xlabel('number of concepts a neu responds to')
plt.ylabel('probability')
plt.legend()
plt.grid()
plt.yscale('log')
plt.savefig('Figures/Data_number_of_concepts_a_neu_responds_to_all_pyramidal_and_interneus.pdf')
plt.show()
print( 'tot recorded neurons = ', np.sum(number_concepts_neus_respond_to))
'''

# -----------------  average of probs over all sessions weighted on the number of neurons in each session -----------------------------------
print('used neus = ', (1. - prob_strict_parent_pattern(64, 0))*100000)


f = open("../Files/matrix_of_group_sizes_times_sections.dat", "w")
f1 = open("../Files/neuron_that_respond_per_session.dat", "w")
number_concepts_neus_respond_to_parent = np.zeros(41)
number_concepts_neus_respond_to_strict_parent = np.zeros(41)
number_concepts_neus_respond_to_random = np.zeros(41)
number_concepts_neus_respond_to_data = np.zeros(41)
prefactor_data = float(TOT_NEUS) /(TOT_NEUS - TOT_NEUS*full_data_vector[0])

print('prefactor_data =', prefactor_data, TOT_NEUS*full_data_vector[0])
n_tot_goups = 0
n_tot_neus = 0
sum_fraction_of_neus = 0
for sess in range(100):
    print('responses of neuron 1 = ',np.shape(responses[0,sess][1,:])[0])
    
    N_neu = np.shape(responses[0,sess][:,0])[0]
    #print(np.shape(responses[0,sess][:,0])[0], np.shape(responses[0,sess][0,:])[0])
    #print('SSSSSSUM ', np.sum(responses[0,sess]))
    for i in range(np.shape(responses[0,sess][:,0])[0]):
        if np.sum(responses[0,sess][i,:]) == 0:
                N_neu -= 1.
    #print('final number of neurons in this session = ', N_neu)
    f1.write('%s   '%N_neu)
    f1.write('\n')
    fraction_of_neus = float(N_neu)/(TOT_NEUS - TOT_NEUS*full_data_vector[0])
    print('fraction_of_neus', fraction_of_neus)
    sum_fraction_of_neus += fraction_of_neus
    N_stim, distance_matrix, mean = Distances_matrix(responses, sess)
    #print(N_stim, distance_matrix, mean)
    threshold = mean #(1.+ mean)/2.
    print('------------------- session ', sess, '----------------------')
    if N_stim == 0 or N_neu == 0:
            print('############### EMPTY session ', sess, '#############')
    elif N_stim != 0 and N_neu != 0:
        # this function is for hierarchical clusering
        model = AgglomerativeClustering(affinity='precomputed', n_clusters=None, compute_full_tree = True, distance_threshold= threshold, linkage='complete').fit(distance_matrix)
        #print(model.labels_)
        #dictionary of counts: Key = counted number, argument = number of counts
        counts = collections.Counter(model.labels_)
        n_groups = len(counts)
        n_tot_goups += n_groups
        print('n_groups =', n_groups)
        stims_groups = []
        for i in range(n_groups):
            stims_groups += [counts[i]]
        #print('stims_groups = ', stims_groups) #so far so good here
        #sorted_goups = sorted(stims_groups)
        max_j = np.max(stims_groups)
        #print('max_j ', max_j)
        groups = [0]*(max_j+1) # in entry the number of group which size is the entry: example groups[1] = # of groups of size 1
        for i in range(max_j+1):
            for j in stims_groups:
                if j==i:
                    groups[i] += 1
        #print('groups =', groups)
        for i in range(max_j+1):
            f.write('%s   '%groups[i])
        for i in range(max_j+1, 45):
            f.write('%s   '%0)
        f.write('\n')
        n_concepts_neus_respond_to_parent, n_concepts_neus_respond_to_strict_parent, n_concepts_neus_respond_to_random = Number_concepts_neus_respond_to(max_j, groups) # I get the probabilities of each group (normalized over the group) and I sum them over groups
        #print('#####  n_concepts_neus_respond_to_parent = ', n_concepts_neus_respond_to_parent)
        #lenght = len(n_concepts_neus_respond_to_parent)
        
        #print('check :', np.min(len(n_concepts_neus_respond_to_parent), 30))
        for j in range(np.min([max_j, 41])):
            #print(j, number_concepts_neus_respond_to_parent[j], n_concepts_neus_respond_to_parent[j])
            number_concepts_neus_respond_to_parent[j] += n_concepts_neus_respond_to_parent[j]*fraction_of_neus
            number_concepts_neus_respond_to_strict_parent[j] += n_concepts_neus_respond_to_strict_parent[j]*fraction_of_neus
            number_concepts_neus_respond_to_random[j] += n_concepts_neus_respond_to_random[j]*fraction_of_neus
    else:
        print('no stim!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        
normalization = np.sum(number_concepts_neus_respond_to_data)
number_concepts_neus_respond_to_data = number_concepts_neus_respond_to_data/float(normalization)
print(np.sum(number_concepts_neus_respond_to_data))
print('sum_fraction_of_neus = ', sum_fraction_of_neus)
f.close()
f1.close()
x = np.linspace(1, 41, 41)
plt.figure()
plt.plot(x,number_concepts_neus_respond_to_parent, 'bs-', label= 'indicator n')
plt.plot(x,number_concepts_neus_respond_to_strict_parent, 'co-', label= 'hierarchical gen')
plt.plot(x,number_concepts_neus_respond_to_random, 'r^--', label= 'itarative')
y = np.linspace(1, len(full_data_vector)-1, len(full_data_vector)-1)
rescaled_data = prefactor_data*np.array(full_data_vector[1:len(full_data_vector)])
plt.plot(y,rescaled_data , 'g*', label = 'data')

print('indicator n = ', number_concepts_neus_respond_to_parent)
print('indicator n = ', number_concepts_neus_respond_to_strict_parent)
print( 'iterative = ', number_concepts_neus_respond_to_random)
print('this must be one =',np.sum(number_concepts_neus_respond_to_parent), np.sum(number_concepts_neus_respond_to_strict_parent),  np.sum(number_concepts_neus_respond_to_random), np.sum(rescaled_data))

plt.plot(x,number_concepts_neus_respond_to_data , 'm*', label = 'data')
"""
plt.plot(y[0:10],rescaled_data[0:10] , 'g*', label = 'data')
bin1 = np.sum(rescaled_data[10:13])/3.
plt.plot(y[10:13],[bin1, bin1, bin1] , 'g*')
bin2 = np.sum(rescaled_data[13:16])/3.
plt.plot(y[13:16],[bin2, bin2, bin2] , 'g*')
bin3 = np.sum(rescaled_data[16:19])/3.
plt.plot(y[16:19],[bin3, bin3, bin3] , 'g*')
bin4 = np.sum(rescaled_data[19:22])/3.
plt.plot(y[19:22],[bin4, bin4, bin4] , 'g*')
bin5 = np.sum(rescaled_data[22:25])/3.
plt.plot(y[22:25],[bin5, bin5, bin5] , 'g*')
plt.plot(y[25:len(full_data_vector)],[0, 0, 0, 0] , 'g*')
"""
plt.xlabel('number of concepts a neu responds to')
plt.ylabel('probability')
plt.yscale('log')
plt.ylim([10**(-7), 2])
plt.legend()
plt.grid()
plt.savefig('Figures/mean_of_sessions_gamma%s_c%s.pdf'%(gamma, c))
plt.show()
print('number of responsive neurons  = ', (TOT_NEUS - TOT_NEUS*full_data_vector[0]))



