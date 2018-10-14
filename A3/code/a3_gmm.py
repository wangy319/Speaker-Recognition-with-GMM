from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import os, fnmatch
import random
import multiprocessing as mp
from scipy.special import logsumexp

# dataDir = '/u/cs401/A3/data/'
dataDir = "/Users/yingxue_wang/Documents/cdf/csc401/Assignment3/A3/data"

class theta:
    def __init__(self, name, M=8, d=13 ):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

        # self.name = name
        # self.omega = 1/M*np.ones((M,1))
        # # self.mu = np.zeros((M,d))           # d variables for M samples
        # self.mu = np.tile(np.array([[ -7.82687105, -15.72149868,  -4.97102738,  -1.28394684,
        #  -0.34701377,  -5.22997612,  -7.99557349,  -6.75920808,
        #   0.77549396, -15.74811886,   0.87531007,  -5.45048304,
        # -13.3664083 ]]),(M,1))
        # self.Sigma = 1/M*np.ones((M,d))     # d variables for M samples
        #

def log_b_m_x(m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'M' that applies to all X outside of this function.
        If you do this, you pass that precomputed component in preComputedForM
    '''
    # Getting Constant M and d
    M, d = myTheta.Sigma.shape

    term1 = np.sum(np.square(x-myTheta.mu[m])/myTheta.Sigma[m],axis = 1)
    term2 = d/2 * np.log(2*np.pi)
    term3 = 1/2 * np.sum(np.log(np.square(myTheta.Sigma[m])))
    return  - term1 - term2 - term3      # shape (8,)

def log_p_m_x(m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''

    # Getting Constant M and d
    M, d = myTheta.Sigma.shape

    # parallel processing
    with mp.Pool(processes=min(mp.cpu_count(), M)) as pool:
        log_bs = [pool.apply_async(log_b_m_x, args=(i, x, myTheta)) for i in range(M)]
        log_Bs = [b.get() for b in log_bs]      # Array of collection of log_bs

    log_Bs = np.array(log_Bs)
    term1 = np.log(myTheta.omega[m,0])
    term2 = log_Bs[m]
    term3 = logsumexp(np.log(myTheta.omega[:, 0]) + log_Bs)

    return term1 + term2 - term3

def logLik(log_Bs, myTheta):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x
        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''

    # log_Ws = np.log(myTheta.omega[:])
    log_Ws = np.log(myTheta.omega)
    log_Ps = logsumexp(log_Bs + log_Ws, axis=0)

    return np.sum(log_Ps)


def get_log_Bs(myTheta, X, M, T, d):
    '''
    :return: Vectorized lod_Bs
    '''

    log_Bs = np.zeros((M, T))
    for m in range(M):
        term1 = np.sum(np.square(X - myTheta.mu[m]) / myTheta.Sigma[m], axis=1)
        term2 = d / 2 * np.log(2 * np.pi)
        term3 = 1. / 2 * np.sum(np.log(np.square(np.prod(myTheta.Sigma[m]))))
        log_bs = - term1 - term2 - term3
        log_Bs[m] = log_bs
    return log_Bs


def train(speaker, X, M = 8, epsilon = 0.0, maxIter = 20):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''

    # Initialize myTheta
    T,d = X.shape
    myTheta = theta(speaker, M, d)

    # Defining constants
    i = 0
    prev_L = float('-inf')
    improvement = float('inf')

    # Initializing theta
    myTheta.Sigma[:, :] = 1.
    indices = np.random.choice(X.shape[0], M, replace=False)
    myTheta.mu = X[indices]
    myTheta.omega[:, 0] = 1. / M

    while(i <= maxIter and improvement > epsilon):
        # Compute Intermediate Result
        # Euqation 1, log_b_m_x, save to ndarray log_Bs
        log_Bs = get_log_Bs(myTheta, X, M, T, d)
        # Euqation 2, log_p_m_x, save to ndarray log_Ps
        log_WBs = log_Bs + np.log(myTheta.omega)
        log_Ps = log_WBs - logsumexp(log_WBs, axis=0)

        # likelihood
        L = logLik(log_Bs, myTheta)
        # Update Parameters
        for m in range(M):
            # log of sum of p
            P_m = np.exp(log_Ps[m])
            P_sum_m = np.sum(P_m)
            # omega
            myTheta.omega[m] = P_sum_m / T
            # mu
            myTheta.mu[m] = np.dot(P_m, X) / P_sum_m
            # sigma
            Sigma_m = (np.dot(P_m, np.square(X)) / P_sum_m) - (myTheta.mu[m] ** 2)
            myTheta.Sigma[m] = Sigma_m

        improvement = L - prev_L
        prev_L = L
        i += 1

    # List for precomputed data for M, needed to be passed into logLik
    preComputedForM = []
    return myTheta


def test(mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''

    models_likelihood = []
    M = models[0].omega.shape[0]
    T = mfcc.shape[0]
    d = mfcc.shape[1]

    bestModel = -1
    bestLogLik = float('-inf')

    for i in range(len(models)):
        theta = models[i]
        log_Bs = get_log_Bs(theta, mfcc, M, T, d)
        log_Lik = logLik(log_Bs, theta)
        models_likelihood.append((theta,log_Lik))

        # best theta
        if(log_Lik > bestLogLik):
            bestLogLik = log_Lik
            bestModel = i

    # Find best k models
    models_likelihood.sort(key=lambda x: x[1], reverse=True)
    print(models[correctID].name)

    for j in range(k):
        print(models_likelihood[j][0].name, models_likelihood[j][1])
    print('\n')

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    random.seed(0)

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    print('TODO: Please refer to file gmm_experiment.py for detailed parameter tuning for Sec 2.3')

    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20

    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate 
    numCorrect = 0
    for i in range(0,len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)

    accuracy = 1.0 * numCorrect/len(testMFCCs)

    print(accuracy)
