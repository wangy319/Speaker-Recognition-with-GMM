from a3_gmm import *
from sklearn.decomposition import PCA

dataDir = '/u/cs401/A3/data/'
# dataDir = "/Users/yingxue_wang/Documents/cdf/csc401/Assignment3/A3/data"

if __name__ == "__main__":

    random.seed(0)

    trainThetas = []
    testMFCCs = []
    print('TODO: Please refer to file gmm_experiment.py for detailed parameter tuning for Sec 2.3')

    d = 13
    d_r_list = [10, 8, 7, 6, 5, 4, 3, 2]
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    maxSpeaker = 32

    for d_r in d_r_list:
        X_pca = np.empty((0, d))
        for subdir, dirs, files in os.walk(dataDir):
            for speaker in dirs:
                files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
                random.shuffle(files)

                for file in files:
                    myMFCC = np.load(os.path.join(dataDir, speaker, file))
                    X_pca = np.append(X_pca, myMFCC, axis=0)

        # fit global PCA
        pca = PCA(n_components=pca_d)
        pca.fit(X_pca)
        np.save('pca_{}.npy'.format(pca_d), pca.components_)
        f = open('pca_%d.txt' % d_r, 'w')

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
                X = pca.transform(X)

                trainThetas.append(train(speaker, X, M, epsilon, maxIter))

               # evaluate
        numCorrect = 0
        for i in range(0, len(testMFCCs)):
            X = testMFCCs[i]
            X = pca.transform(X)
            numCorrect += test(X, i, trainThetas, k)

        accuracy = 1.0 * numCorrect / len(testMFCCs)

        print(accuracy)
        print('pca_dim: {} \t M: {} \t maxIter: {} \t S: {} \t Accuracy: {}'.format(d_r, M, maxIter, maxSpeaker, accuracy), file=f)

