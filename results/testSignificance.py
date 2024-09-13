# This script has been extracted from the codebase of the paper The Hitchhiker’s Guide to Testing 
# Statistical Significance in NaturalLanguage Processing, by Rotem Dror, Gili Baumer, Segev Shlomov 
# and Roi Reichart. The original codebase can be found at https://github.com/rtmdrr/testSignificanceNLP.git

import sys
import numpy as np
from scipy import stats




### Normality Check
# H0: data is normally distributed
def normality_check(data_A, data_B, name, alpha):

    if(name=="Shapiro-Wilk"):
        # Shapiro-Wilk: Perform the Shapiro-Wilk test for normality.
        shapiro_results = stats.shapiro([a - b for a, b in zip(data_A, data_B)])
        return shapiro_results[1]

    elif(name=="Anderson-Darling"):
        # Anderson-Darling: Anderson-Darling test for data coming from a particular distribution
        anderson_results = stats.anderson([a - b for a, b in zip(data_A, data_B)], 'norm')
        sig_level = 2
        if(float(alpha) <= 0.01):
            sig_level = 4
        elif(float(alpha)>0.01 and float(alpha)<=0.025):
            sig_level = 3
        elif(float(alpha)>0.025 and float(alpha)<=0.05):
            sig_level = 2
        elif(float(alpha)>0.05 and float(alpha)<=0.1):
            sig_level = 1
        else:
            sig_level = 0

        return anderson_results[1][sig_level]

    else:
        # Kolmogorov-Smirnov: Perform the Kolmogorov-Smirnov test for goodness of fit.
        ks_results = stats.kstest([a - b for a, b in zip(data_A, data_B)], 'norm')
        return ks_results[1]

## McNemar test
def calculateContingency(data_A, data_B, n):
    ABrr = 0
    ABrw = 0
    ABwr = 0
    ABww = 0
    for i in range(0,n):
        if(data_A[i]==1 and data_B[i]==1):
            ABrr = ABrr+1
        if (data_A[i] == 1 and data_B[i] == 0):
            ABrw = ABrw + 1
        if (data_A[i] == 0 and data_B[i] == 1):
            ABwr = ABwr + 1
        else:
            ABww = ABww + 1
    return np.array([[ABrr, ABrw], [ABwr, ABww]])

def mcNemar(table):
    statistic = float(np.abs(table[0][1]-table[1][0]))**2/(table[1][0]+table[0][1])
    pval = 1-stats.chi2.cdf(statistic,1)
    return pval


#Permutation-randomization
#Repeat R times: randomly flip each m_i(A),m_i(B) between A and B with probability 0.5, calculate delta(A,B).
# let r be the number of times that delta(A,B)<orig_delta(A,B)
# significance level: (r+1)/(R+1)
# Assume that larger value (metric) is better 
def rand_permutation(data_A, data_B, n, R):
    delta_orig = float(sum([ x - y for x, y in zip(data_A, data_B)]))/n
    r = 0
    for x in range(0, R):
        temp_A = data_A
        temp_B = data_B
        samples = [np.random.randint(1, 3) for i in xrange(n)] #which samples to swap without repetitions
        swap_ind = [i for i, val in enumerate(samples) if val == 1]
        for ind in swap_ind:
            temp_B[ind], temp_A[ind] = temp_A[ind], temp_B[ind]
        delta = float(sum([ x - y for x, y in zip(temp_A, temp_B)]))/n
        if(delta<=delta_orig):
            r = r+1
    pval = float(r+1.0)/(R+1.0)
    return pval


#Bootstrap
#Repeat R times: randomly create new samples from the data with repetitions, calculate delta(A,B).
# let r be the number of times that delta(A,B)<2*orig_delta(A,B). significance level: r/R
# This implementation follows the description in Berg-Kirkpatrick et al. (2012), 
# "An Empirical Investigation of Statistical Significance in NLP".
def Bootstrap(data_A, data_B, n, R):
    delta_orig = float(sum([x - y for x, y in zip(data_A, data_B)])) / n
    r = 0
    for x in range(0, R):
        temp_A = []
        temp_B = []
        samples = np.random.randint(0,n,n) #which samples to add to the subsample with repetitions
        for samp in samples:
            temp_A.append(data_A[samp])
            temp_B.append(data_B[samp])
        delta = float(sum([x - y for x, y in zip(temp_A, temp_B)])) / n
        if (delta > 2*delta_orig):
            r = r + 1
    pval = float(r)/(R)
    return pval