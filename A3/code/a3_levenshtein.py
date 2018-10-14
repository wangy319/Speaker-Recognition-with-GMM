import os
import numpy as np
import re
import string
from scipy import stats

dataDir = '/u/cs401/A3/data/'
# dataDir = "/Users/yingxue_wang/Documents/cdf/csc401/Assignment3/A3/data"

punctuations = re.compile('[%s]' % re.escape(string.punctuation.replace('[', '').replace(']', '')))
labels = re.compile(r'<\w+>|\[\w+\]| \w+/\w+:\w+ ')

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """

    n = len(r)
    m = len(h)
    R = np.zeros((n+1, m+1))
    B = np.zeros((n+1, m+1, 3))

    # Setting Matrix of Distance
    for i in range(1,n+1):
        R[i,0] = float('inf')
    for j in range(1,m+1):
        R[0,j] = float('inf')

    for i in range(1,n+1):
        for j in range(1,m+1):
            deletion = R[i-1,j] + 1
            substitution = R[i-1, j-1] + 0 if r[i-1] == h[j-1] else R[i-1, j-1] + 1
            insertion = R[i,j-1] + 1
            R[i,j] = min(deletion, substitution, insertion)

            # If Deletion
            if R[i,j] == R[i-1,j] + 1:
                B[i, j, 0] = B[i - 1, j, 0] + 1  # Up
                B[i, j, 1] = B[i - 1, j, 1]
                B[i, j, 2] = B[i - 1, j, 2]
            # If insertion
            elif R[i, j] == R[i, j-1] + 1:
                B[i, j, 0] = B[i, j - 1, 0]
                B[i, j, 1] = B[i, j - 1, 1] + 1 # left
                B[i, j, 2] = B[i, j - 1, 2]
            # If substitution
            else:
                B[i, j, 0] = B[i - 1, j - 1, 0]
                B[i, j, 1] = B[i - 1, j - 1, 1]
                B[i, j, 2] = B[i-1, j-1, 2] + 0 if r[i-1] == h[j-1] else B[i-1, j-1, 2] + 1

    counts = B[-1,-1,:]
    # results: wer, subs, ins, dels
    return R[n,m]/n , counts[2], counts[1], counts[0]

def preprocess_line(lines):
    lines = lines.strip()
    lines = labels.sub('', lines)
    lines = punctuations.sub('', lines)
    tokens = list(filter(lambda y: len(y) > 0, map(lambda x: x.strip().lower(), lines.split())))
    return tokens

if __name__ == "__main__":

    Google_WER = []
    Kaldi_WER = []

    f = open('asrDiscussion.txt', 'w')

    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            # read files without newlines characters
            path = os.path.join(dataDir, speaker)
            reference = open(path + '/transcripts.txt', 'r').read().splitlines()
            google = open(path + '/transcripts.Google.txt', 'r').read().splitlines()
            kaldi = open(path + '/transcripts.Kaldi.txt', 'r').read().splitlines()

            print(speaker)

            if len(reference) * len(google) * len(kaldi) > 0:
                for i in range(min(len(reference), len(google), len(kaldi))):
                    reference_line = preprocess_line(reference[i])

                    # Google
                    wer, subs, ins, dels = Levenshtein(reference_line, preprocess_line(google[i]))
                    print('{} {} {} {} S:{}, I:{}, D:{}'.format(speaker, 'Google', i, wer, subs, ins, dels), file=f)
                    Google_WER.append(wer)

                    # Kaldi
                    wer, subs, ins, dels = Levenshtein(reference_line, preprocess_line(kaldi[i]))
                    print('{} {} {} {} S:{}, I:{}, D:{}'.format(speaker, 'Kaldi', i, wer, subs, ins, dels), file=f)
                    Kaldi_WER.append(wer)

                print('\n', file=f)

    # summary
    google_avg, google_std = np.mean(Google_WER), np.std(Google_WER)
    kaldi_avg, kaldi_std = np.mean(Kaldi_WER), np.std(Kaldi_WER)

    print('Google average: {}. Google standard deviation: {}. Kaldi average: {}. '
          'Kaldi standard deviation: {}. Statistical significance (p-value): {}.'
          .format(google_avg, google_std, kaldi_avg, kaldi_std, stats.ttest_ind(Google_WER, Kaldi_WER)), file=f)
    f.close()

