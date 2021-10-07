'''This script is used to generate and store the accuracy obtained for each tweet 
by running the HawkEye with different combinations of weighing constants and parameters'''

import pandas as pd
import numpy as np
import math
import pickle
import itertools

def rev2(ratings,notes,lambda1,lambda2,lambda3,init_goodness,convergence_threshold,alpha1,beta1,gamma1,delta1,mu_r,mu_w,mu_t,mu_g):
 
    #do initializations
    ratings['goodness'] = [init_goodness]*len(ratings)
    ratings['rating'] = ratings.apply(lambda x : 1 if x['helpful']==1 else -1,axis=1)
    notes['goodness'] = [init_goodness]*len(notes)
    notes['verdict'] = notes.apply(lambda x : 1 if x['classification']=='NOT_MISLEADING' else -1,axis=1)

    #DO
    #Fairness of user in rating notes
    ratings['score_goodness_difference_metric'] = 1-((ratings['rating']-ratings['goodness']).abs()/2)
    ratings['rating_fairness'] = (ratings.groupby(['participantId'])['score_goodness_difference_metric'].transform("sum") + alpha1*mu_r)/(ratings.groupby(['participantId'])['participantId'].transform("count") + alpha1)
    
    #Fairness of user in writing notes
    notes['writing_fairness'] = (notes.groupby(['participantId'])['goodness'].transform("sum") + beta1*mu_w)/(notes.groupby(['participantId'])['participantId'].transform("count") + beta1)
    
    #Accuracy of Tweet
    notes['weighted_goodness'] = notes['goodness']*notes['verdict']
    notes['tweet_accuracy'] = (notes.groupby(['tweetId'])['weighted_goodness'].transform("sum") + delta1*mu_t)/(notes.groupby(['tweetId'])['tweetId'].transform("count") + delta1)
    
    #Goodness of notes
    ratings['weighted_rating_fairness'] = ratings['rating_fairness']*ratings['rating']
    ratings['goodness_term1'] = (ratings.groupby(['noteId'])['weighted_rating_fairness'].transform("sum") + gamma1*mu_g)/(ratings.groupby(['noteId'])['noteId'].transform("count") + gamma1)
    notes['goodness_term1'] = lambda1*notes.apply(lambda x: 1 if len(ratings.loc[ratings['noteId'] == x['noteId']])==0 else ratings.loc[ratings['noteId'] == x['noteId']].iloc[0]['goodness_term1'],axis=1)
    notes['goodness_term3'] = lambda3*(1-(notes['tweet_accuracy']-notes['verdict']).abs())
    notes['goodness'] = 1/3 * (notes['goodness_term1'] + lambda2*notes['writing_fairness'] + notes['goodness_term3'])
    
    #IMPORTANT : Update goodness ratings df
    ratings['goodness'] = ratings.apply(lambda x: notes.loc[notes['noteId'] == x['noteId']].iloc[0]['goodness'],axis=1)

    #WHILE
    t = 1
    error = math.inf

    while(error>convergence_threshold):

        old_rating_fairness_values = np.array(ratings['rating_fairness'])
        old_writing_fairness_values = np.array(notes['writing_fairness'])
        old_tweet_accuracy_values = np.array(notes['tweet_accuracy'])
        old_goodness_values = np.array(notes['goodness'])

        #Fairness of user in rating notes
        ratings['score_goodness_difference_metric'] = 1-((ratings['rating']-ratings['goodness']).abs()/2)
        ratings['rating_fairness'] = (ratings.groupby(['participantId'])['score_goodness_difference_metric'].transform("sum") + alpha1*mu_r)/(ratings.groupby(['participantId'])['participantId'].transform("count") + alpha1)
        
        #Fairness of user in writing notes
        notes['writing_fairness'] = (notes.groupby(['participantId'])['goodness'].transform("sum") + beta1*mu_w)/(notes.groupby(['participantId'])['participantId'].transform("count") + beta1)
        
        #Accuracy of Tweet
        notes['weighted_goodness'] = notes['goodness']*notes['verdict']
        notes['tweet_accuracy'] = (notes.groupby(['tweetId'])['weighted_goodness'].transform("sum") + delta1*mu_t)/(notes.groupby(['tweetId'])['tweetId'].transform("count") + delta1)
        
        #Goodness of notes
        ratings['weighted_rating_fairness'] = ratings['rating_fairness']*ratings['rating']
        ratings['goodness_term1'] = (ratings.groupby(['noteId'])['weighted_rating_fairness'].transform("sum") + gamma1*mu_g)/(ratings.groupby(['noteId'])['noteId'].transform("count") + gamma1)
        notes['goodness_term1'] = lambda1*notes.apply(lambda x: 1 if len(ratings.loc[ratings['noteId'] == x['noteId']])==0 else ratings.loc[ratings['noteId'] == x['noteId']].iloc[0]['goodness_term1'],axis=1)
        notes['goodness_term3'] = lambda3*(1-(notes['tweet_accuracy']-notes['verdict']).abs())
        notes['goodness'] = 1/3 * (notes['goodness_term1'] + lambda2*notes['writing_fairness'] + notes['goodness_term3'])
        
        #IMPORTANT : Update goodness ratings df
        ratings['goodness'] = ratings.apply(lambda x: notes.loc[notes['noteId'] == x['noteId']].iloc[0]['goodness'],axis=1)

        new_rating_fairness_values = np.array(ratings['rating_fairness'])
        new_writing_fairness_values = np.array(notes['writing_fairness'])
        new_tweet_accuracy_values = np.array(notes['tweet_accuracy'])
        new_goodness_values = np.array(notes['goodness'])

        rating_fairness_error = np.sum(np.absolute((np.subtract(old_rating_fairness_values,new_rating_fairness_values))))
        writing_fairness_error = np.sum(np.absolute(np.subtract(old_writing_fairness_values,new_writing_fairness_values)))
        tweet_accuracy_error = np.sum(np.absolute(np.subtract(old_tweet_accuracy_values,new_tweet_accuracy_values)))
        goodness_error = np.sum(np.absolute(np.subtract(old_goodness_values,new_goodness_values)))

        error = max(rating_fairness_error,writing_fairness_error,tweet_accuracy_error,goodness_error)
        t += 1
        
    return notes,ratings

def runRev2(results,parameterCombination,keywordArgs):
    
    ratings = keywordArgs['ratingsGlobal']
    notes= keywordArgs['notesGlobal']
    init_goodness = keywordArgs['init_goodness']
    convergence_threshold = keywordArgs['convergence_threshold']
    
    mu_r = keywordArgs['mu_r']
    mu_w = keywordArgs['mu_w']
    mu_t = keywordArgs['mu_t']
    mu_g = keywordArgs['mu_g']
    
    alpha1 = parameterCombination[0]
    beta1 = parameterCombination[1]
    gamma1 = parameterCombination[2]
    delta1 = parameterCombination[3]
    
    lambda1 = parameterCombination[4]
    lambda2 = parameterCombination[5]
    lambda3 = parameterCombination[6]
    
    notes_new,ratings_new = rev2(ratings,notes,lambda1,lambda2,lambda3,init_goodness,convergence_threshold,alpha1,beta1,gamma1,delta1,mu_r,mu_w,mu_t,mu_g)

    results[parameterCombination] = (notes_new,ratings_new)
    return results

if __name__ == '__main__':
    
    init_goodness = 1
    convergence_threshold = 0.001
    
    notesGlobal = pd.read_csv("..//data//notes-00000-13-04-21.tsv", sep='\t')
    ratingsGlobal = pd.read_csv("..//data//ratings-00000-13-04-21.tsv", sep='\t')
    notesGlobal = notesGlobal[['noteId', 'participantId','tweetId','classification']]
    ratingsGlobal = ratingsGlobal[['noteId', 'participantId','helpful','notHelpful']]
    
    no_of_rating_participants = len(set(ratingsGlobal['participantId']))
    no_of_writing_participants = len(set(notesGlobal['participantId']))
    no_of_tweets = len(set(notesGlobal['tweetId']))
    no_of_notes = len(set(notesGlobal['noteId']))
    mu_r = 1*no_of_rating_participants/no_of_rating_participants
    mu_w = 1*no_of_writing_participants/no_of_writing_participants
    mu_t = 1*no_of_tweets/no_of_tweets                                       
    mu_g = 1*no_of_notes/no_of_notes
    
    keywordArgs = {'ratingsGlobal': ratingsGlobal,
                    'notesGlobal' : notesGlobal,
                    'init_goodness' : init_goodness,
                    'convergence_threshold' : convergence_threshold,
                    'mu_r' : mu_r,
                    'mu_w' : mu_w,
                    'mu_t' : mu_t,
                    'mu_g' : mu_g}
    
    a = [0,1,2] #CHANGE THIS TO CHANGE/ADD VALUES OF WEIGHING CONSTANTS
    b = [0,0.5,1] #CHANGE THIS TO CHANGE/ADD VALUES OF SMOOTHING PARAMETERS
    parameterCombinations = list(itertools.product(a,a,a,a,b,b,b))

    results = {}
    for parameterCombination in parameterCombinations:    
        results = runRev2(results,parameterCombination,keywordArgs)
        
    results = runRev2(results,parameterCombination,keywordArgs)
    results_dict = dict(results)
    
    with open('results/hawkeye_all_combination_parameter_runs_result.pickle', 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
