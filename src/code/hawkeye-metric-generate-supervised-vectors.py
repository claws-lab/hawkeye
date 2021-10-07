'''This script is used to generate the vectors, which will be later used to train a model
to detect misinformation tweets. Each tweet is represetned as a feature vector of its 
accuracy scores across multiple runs of HawkEye for different combinations of values 
of the weighing constants and of the smoothing parameters'''

import pandas as pd
import pickle5
import itertools
from tqdm import tqdm

'''Read this file stores the accuracy obtained for each tweet by running the HawkEye 
with different combinations of weighing constants and parameters'''
with open('results/hawkeye_all_combination_parameter_runs_result.pickle', 'rb') as handle:
    results = dict(pickle5.load(handle))
        
'''These paramters and constants must be the same as those used in 
run-hawkeye-for-different-parameter-combinations.py script'''
a = [0,1,2]
b = [0,0.5,1]
parameterCombinations = list(itertools.product(a,a,a,a,b,b,b))

notesGlobal = pd.read_csv("..//data//notes-00000-13-04-21.tsv", sep='\t')
ratingsGlobal = pd.read_csv("..//data//ratings-00000-13-04-21.tsv", sep='\t')
notesGlobal = notesGlobal[['noteId', 'participantId','tweetId','classification']]
ratingsGlobal = ratingsGlobal[['noteId', 'participantId','helpful','notHelpful']]

totalTweets = list(set(notesGlobal['tweetId']))
tweet_vectors = {}
for tweetId in tqdm(totalTweets):
    tweet_vector = []
    for parameterCombination in parameterCombinations:
        notes, ratings = results[parameterCombination]
        tweet_vector.append(notes.loc[notes['tweetId'] == tweetId]['tweet_accuracy'].iloc[0])
        
    tweet_vectors[tweetId] = tweet_vector 
          
with open('results/hawkeye_metric_tweet_vectors_supervised.pickle', 'wb') as handle:
    pickle5.dump(tweet_vectors, handle)