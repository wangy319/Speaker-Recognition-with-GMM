from a3_gmm import *

if __name__ == "__main__":

    random.seed(0)

    trainThetas = []
    testMFCCs = []

    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    epsilon = 0.0
    maxIter = 20
    M = 8
    # maxSpeaker = 32
    f = open('gmm_S.txt', 'w')

    M_list = [8, 7, 6, 5, 4, 3, 2, 1]
    maxIter_list = [20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0]
    maxSpeaker_list = [32, 24, 16, 8, 4]

    # for M in M_list:
    # for maxIter in maxIter_list:
    for maxSpeaker in maxSpeaker_list:
        s = 0
        trainThetas = []
        testMFCCs = []
        for subdir, dirs, files in os.walk(dataDir):
            for speaker in dirs:
                print(speaker)

                files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
                random.shuffle(files)

                if s < maxSpeaker:
                    testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
                    testMFCCs.append(testMFCC)

                # if s < maxSpeaker:
                    X = np.empty((0, d))
                    for file in files:
                        myMFCC = np.load(os.path.join(dataDir, speaker, file))
                        X = np.append(X, myMFCC, axis=0)

                    trainThetas.append(train(speaker, X, M, epsilon, maxIter))
                else:
                    trainThetas.append(theta(speaker))
                s += 1

        # evaluate
        numCorrect = 0
        for i in range(0, len(testMFCCs)):
            numCorrect += test(testMFCCs[i], i, trainThetas, k)

        accuracy = 1.0 * numCorrect / len(testMFCCs)
        print('Accuracy: ', accuracy)
        print('M: {} \t maxIter: {} \t S: {} \t Accuracy: {}'.format(M, maxIter, maxSpeaker, accuracy))
        print('\n')

        # write to file
        print('M: {} \t maxIter: {} \t S: {} \t Accuracy: {}'.format(M, maxIter, maxSpeaker, accuracy), file=f)
