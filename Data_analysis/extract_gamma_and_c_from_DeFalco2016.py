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
    8) 'familiarity': in each trial, the  N_stim x N_stim  matrix ???
    9) 'stimulus_name': in each trial, the  N_stim x 1 with the stimuli name, same as stimuli.a
    10) 'stimulus_category': in each trial, the  N_stim x 1 with the stimuli category, same as stimuli.c
    11) 'stimulus_category_code': in each trial, the  N_stim x 1 with the stimuli category code, same as stimuli.d
    12) 'neuron_classification': in each trial, the  N_neus x 1, with the neuron classification into 'unknown', 'pyramidal', 'interneuron'.
    13) 'Avg_Fr': : in each trial, the  N_neus x 1 ???
    14) 'mean_spikeshape': in each trial, the  N_neus x 1 ???
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

c = 0.04
# there is a dependence on c, for c = 0.2 the parent pattern distribution stabilizes at 10^-5 instead of 10^-6
gamma = 0.002  #0.002 #have to be bigger than c
C = (c - gamma)/(1. - gamma)

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

def S_parent(pmax):
    return 1 - prob_parent_pattern(pmax, 0)

def S_random(pmax):
    return 1 - prob_random_pattern(pmax, 0)

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
    final_fit_parent_hom = [0]*(pmax)
    final_fit_rnd_hom = [0]*(pmax)
    
    # M is the renormalizaion constant for S
    sum = 0.
    for j in range(1, pmax +1 ):
        #print(j)
        sum += (groups[j] * S_parent(j))
    M = 1. / (sum)

    for  p in range(1, pmax +1):
        for j in range(max(p,1),pmax +1):
            #print('p = ', p, ' and j = ', j)
            final_fit_parent_hom[p-1] += M*groups[j] * prob_parent_pattern(j, p) * S_parent(j)
            final_fit_rnd_hom[p-1] += M*groups[j] * prob_random_pattern(j, p) * S_random(j)

                #print('final_fit_parent_hom[1] = ', final_fit_parent_hom[1])
    factor_p  = np.sum(final_fit_parent_hom)
    factor_r  = np.sum(final_fit_rnd_hom)
    #print('this must be one ', 1./factor_p *np.sum(final_fit_parent_hom),1./factor_r*np.sum(final_fit_rnd_hom))
    return 1./factor_p *np.array(final_fit_parent_hom), 1./factor_r*np.array(final_fit_rnd_hom)


# facciamola strettamente positiva cosi evito problemi con le anticorrelazioni
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
    #print('average of the webassociations = ',np.sum(web)/counts, ' and sigma = ', np.var(web))
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
    #print('average of the final matrix = ',np.average(Modified_associations), ' and sigma = ', np.var(Modified_associations))
    return N_stim, Modified_associations, np.average(Modified_associations)



#----------------------------------------------------------------------------------------------------
##################  MAIN ############################################################################
#----------------------------------------------------------------------------------------------------
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
'''
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
# -----------------  estimate gamma -----------------------------------

n_tot_goups = 0
n_tot_neus = 0
sum_fraction_of_neus = 0
c_final_sess_avg = 0.
c_fin = 0
c_final_group_avg = 0.
tot_n_resp_groups = 0
n_responses = 0.
n_tot_responses = 0
final_frac = 0
for sess in range(100):
    #print(np.shape(responses[0,sess]))
    N_neu = np.shape(responses[0,sess][:,0])[0]
    n_tot_responses += np.sum(responses[0,sess])/(N_neu*np.shape(responses[0,sess][0,:])[0])
    N_stim, distance_matrix, mean = Distances_matrix(responses, sess)
    #print(N_stim, distance_matrix, mean)
    threshold = mean #(1.+ mean)/2.
    #print('------------------- session ', sess, '----------------------')
    if N_stim != 0:
        # this function is for hierarchical clusering
        model = AgglomerativeClustering(affinity='precomputed', n_clusters=None, compute_full_tree = True, distance_threshold= threshold, linkage='complete').fit(distance_matrix)
        #print(model.labels_)
        #print('model.labels_ = ',len(model.labels_))
        #dictionary of counts: Key = counted number, argument = number of counts
        counts = collections.Counter(model.labels_)
        #print(counts)
        n_groups = len(counts)
        
        indices_indip_stim = []
        for g in range(n_groups):
            itemindex = np.where(model.labels_==g)
            indices_indip_stim += [itemindex[0][0]]
            #print(g, 'indices to keep = ', itemindex[0][0])
        #print('indices_indip_stim = ', indices_indip_stim)
        #print('n_groups = ',n_groups, 'len(indices_indip_stim) = ', len(indices_indip_stim))
        indip_stim_matrix = np.zeros([N_neu, len(indices_indip_stim)])
        for i in range(n_groups):
            idx = indices_indip_stim[i]
            indip_stim_matrix[:,i] = responses[0,sess][:,idx]
            
        number_of_resp =  np.zeros(N_neu)
        n_responsive_neus = 0
        c_within_the_sess = 0
        for neu in range(N_neu):
            number_of_resp[neu] = np.sum(indip_stim_matrix[neu,:])
            #print('number_of_resp[%s]'%neu, number_of_resp[neu])
            if number_of_resp[neu] > 0.:
                n_responsive_neus += 1
                #print('number_of_resp[neu] = ', number_of_resp[neu])
                c_local = (number_of_resp[neu] - 1 )/float(n_groups -1)
                #print('### c_local = ', c_local)
                c_within_the_sess += c_local
            
        #print('n_responsive_neus = ', n_responsive_neus)
        if n_responsive_neus > 0 :
            c_within_the_group = c_within_the_sess / float(n_responsive_neus)
            #print('c_within_the_group = ', c_within_the_group)
            c_final_sess_avg += c_within_the_group
            c_final_group_avg += c_within_the_group
            tot_n_resp_groups += 1
            
            
        #second attempt
        avg_p11 = 0.
        counttt = 0
        n_responses += np.sum(indip_stim_matrix)/(N_neu*n_groups)
        frac_shared_neus_sees = 0
        for stim1 in range(len(indices_indip_stim)):
            for stim2 in range(stim1+1,len(indices_indip_stim)):
                #print(np.shape(indip_stim_matrix[neu1,:]), np.shape(indip_stim_matrix[neu2,:]))
                p11 = (indip_stim_matrix[:, stim1]).dot( indip_stim_matrix[:, stim2]) /float(N_neu)
                frac_shared_neus = (indip_stim_matrix[:, stim1]).dot( indip_stim_matrix[:, stim2])/float(N_neu)
                frac_shared_neus_sees += frac_shared_neus
                #print('p11 = ', p11)
                avg_p11 += p11
                counttt += 1
        frac_shared_neus_sees /= float(counttt)
        final_frac += frac_shared_neus_sees
        avg_p11 /= float(counttt)
        c_sess = np.sqrt(avg_p11)
        c_fin += c_sess
        #print('seesion frac and P11', frac_shared_neus_sees, avg_p11 )
print('!!!!!!!!!!!!!!!!!! gamma final from  resp/(N*stim)= ', n_responses/100., n_tot_responses/100.) #, final_frac/100. )#float(responsive_sess)
print('!!!!!!!!!!!!!!!!!! gamma final sqrt(P11) = ', c_fin/100. )#float(responsive_sess)
print('!!!!!!!!!!!!!!!!!! gamma final_sess_avg = ', c_final_sess_avg/100. )#float(responsive_sess))
print('!!!!!!!!!!!!!!!!!! gamma final_group_avg = ', c_final_group_avg/float(tot_n_resp_groups) )#float(responsive_sess))

'''
sum = 0
tot_neu = 0
tot_sim = 0
other_avg = 0
for sess in range(100):
    #print(np.shape(responses[0,sess]))
    N_neu = np.shape(responses[0,sess][:,0])[0]
    N_stim = np.shape(responses[0,sess][0, :])[0]
    tot_neu += N_neu
    tot_sim += N_stim
    sum += np.sum(responses[0,sess])
    gamma_loc = np.sum(responses[0,sess])/float(N_neu + N_stim)
    other_avg += gamma_loc
    print('N_neu', N_neu, 'N_stim', N_stim, 'sum = ', sum, 'gamma_loc = ', gamma_loc)
        
print('different calculation of gamma = ', sum/float(tot_neu + tot_sim) , 'other_avg = ', other_avg/100.)
'''
# -----------------  estimate c -----------------------------------

n_tot_goups = 0
n_tot_neus = 0
sum_fraction_of_neus = 0
c_final_sess_avg = 0.
c_final_group_avg = 0.
tot_n_resp_groups = 0
frac_averaged_sess = 0
responsive_session = 0
for sess in range(100):
    N_neu = np.shape(responses[0,sess][:,0])[0]
    #print('N_neu in this session = ', N_neu)
    N_stim, distance_matrix, mean = Distances_matrix(responses, sess)
    #print(N_stim, distance_matrix, mean)
    threshold = mean #(1.+ mean)/2.
    #print('------------------- session ', sess, '----------------------')
    if N_stim != 0 and N_neu!=0 :
        # this function is for hierarchical clusering
        model = AgglomerativeClustering(affinity='precomputed', n_clusters=None, compute_full_tree = True, distance_threshold= threshold, linkage='complete').fit(distance_matrix)
        #print(model.labels_)
        #print('model.labels_ = ',len(model.labels_))
        #dictionary of counts: Key = counted number, argument = number of counts
        counts = collections.Counter(model.labels_)
        print('counts =', counts)
        n_groups = len(counts)

        print('n_groups',n_groups)
        #indices_same_group = []
        c_averaged_over_grops = 0
        responsive_groups = 0
        frac_averaged_groups = 0
        for g in range(n_groups):
            indices_same_group = np.where(model.labels_==g)
            group_size = len(indices_same_group[0])
            #print('grop_size =', grop_size)
            #print(itemindex[0])
            group_stim_matrix = np.zeros([N_neu, group_size])
            if group_size > 1:
                counttt = 0
                for i in range(group_size):
                    idx = indices_same_group[0][i]
                    #print('idx = ',idx, 'responses[0,sess][:,idx]', responses[0,sess][:,idx])
                    group_stim_matrix[:,i] = responses[0,sess][:,idx]
                for i in range(group_size):
                    for j in range(i+1, group_size):
                        if np.sum(group_stim_matrix[:, i]) > 0:
                            gamma = 0.01
                            #print(group_stim_matrix[:, i])
                            frac_shared_neus_loc = (group_stim_matrix[:, i]).dot( group_stim_matrix[:, j]) /float(N_neu*gamma)
                            #print(i, j, 'frac_shared_neus_loc = ', frac_shared_neus_loc)
                            frac_averaged_groups += frac_shared_neus_loc
                            frac_averaged_sess += frac_shared_neus_loc
                            counttt  += 1
                if counttt>0:
                    frac_averaged_sess /= float(counttt)
                    responsive_session += 1
                #print('group stim matrix sum = ', np.sum(group_stim_matrix))
                number_of_resp =  np.zeros(N_neu)
                n_responsive_neus = 0
                c_within_the_group = 0
                for neu in range(N_neu):
                    number_of_resp[neu] = np.sum(group_stim_matrix[neu,:])
                    #print('number_of_resp[%s]'%neu, number_of_resp[neu])
                    if number_of_resp[neu] > 0.:
                        n_responsive_neus += 1
                        #print('number_of_resp[neu] = ', number_of_resp[neu])
                        c_local = (number_of_resp[neu] - 1 )/float(group_size -1)
                        #print('### c_local = ', c_local)
                        c_within_the_group += c_local
                #print('n_responsive_neus = ', n_responsive_neus)
                if n_responsive_neus > 0 :
                    c_within_the_group = c_within_the_group / float(n_responsive_neus)
                    c_averaged_over_grops += c_within_the_group
                    c_final_group_avg += c_within_the_group
                    responsive_groups += 1
                    tot_n_resp_groups += 1
                    #print('---- c_within_the_group = ', c_within_the_group)
        #print('responsive_groups = ', responsive_groups)
        if responsive_groups != 0 :
            c_averaged_over_grops = c_averaged_over_grops/ float(responsive_groups)
            c_final_sess_avg += c_averaged_over_grops
            frac_averaged_groups /= float(responsive_groups)
            #print('c_averaged_over_grops = ', c_averaged_over_grops)
print('!!!!!!!!!!!!!!!!!! c_final_sess_avg = ', c_final_sess_avg/100.)
print('!!!!!!!!!!!!!!!!!! c_final_group_avg = ', c_final_group_avg/float(tot_n_resp_groups))
print('!!!!!!!!!!!!!!!!!! c_final_group_avg with pairwise associations= ', frac_averaged_sess/float(tot_n_resp_groups), frac_averaged_sess/float(responsive_session))

