import pandas as pd
import numpy as np
import random,string
import multiprocessing
import math
import pickle

def getCurrentTopGoodNotesForTweet(notes,ratings,tweetId,topNotes = 3,minRatingsNeeded = 5,minGoodnessNeeded = 0.75):
    
    ratings_count = ratings.groupby('noteId').count()
    #SELECT THOSE NOTES FROM RATINGS TABLE WHO HAVE ATLEAST minRatingsNeeded RATINGS
    valid_count_notes = set(ratings_count[ratings_count.apply(lambda x : x['participantId'] >= minRatingsNeeded,axis=1)].index)
    #GET ALL NOTES FOR THIS TWEET
    notesForTweet = notes.loc[notes['tweetId'] == tweetId]
    #FROM ALL NOTES SELECT NOTES WHICH WE SHORTLISTED AS HAVING MINIMUM RATINGS
    validCountnotes = notesForTweet.loc[notes['noteId'].isin(valid_count_notes)]
    #FROM THESE NOTES SELECT ONLYTHOSE WHO HAVE THE GOODNESS ABOVE THE THRESHOLD
    filteredNotes = validCountnotes[validCountnotes['goodness'] >= minGoodnessNeeded]
    #RETURN TOP X OF THESE FILTERED NOTES
    return filteredNotes.sort_values(by='goodness', ascending=False)[:topNotes]

def createDummyParticipantId(dummyIdLength,ratings):
    
    while(True):
        dummyParticipantId = "DUMMY" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=dummyIdLength-5))
        if dummyParticipantId not in set(ratings['participantId']):
            break
    return dummyParticipantId

def addDummyRatings(noteId,dummyParticipantIdList,ratingsOld,notes,helpful=1):
    
    numberOfRatings = len(dummyParticipantIdList)
    zeroes,ones,minusOnes = [0]*numberOfRatings,[1]*numberOfRatings,[-1]*numberOfRatings
    
    #ASSIGN SAME GOODNESS TO ALL THE DUMMY RATINGS, SINCE THIS IS FOR THE SAME NOTE
    goodnessList = [notes.loc[notes['noteId'] == noteId]['goodness'].iloc[0]] * numberOfRatings
    
    #ASSIGN SAME GOODNESS TERM 1 TO ALL THE DUMMY RATINGS, SINCE THIS IS FOR THE SAME NOTE    
    goodness_term1List = [notes.loc[notes['noteId'] == noteId]['goodness_term1'].iloc[0]] * numberOfRatings
    
    #SAME NOTEID FOR ALL DUMMY ROWS, SINCE THIS IS FOR THE SAME NOTE
    noteIdTempList = [noteId]*numberOfRatings
    #DUMMY PARTICIPANT IDs
    participantIdTempList = dummyParticipantIdList #important
    
    #IF YOU WANT TO ADD SCORE=1 RATING
    if helpful:
        helpfulTempList = ones  #important
        ratingsList = ones #important
        notHelpfulTempList = zeroes
    #IF YOU WANT TO ADD SCORE=-1 RATING    
    else:
        helpfulTempList = zeroes  #important
        ratingsList = minusOnes #important
        notHelpfulTempList = ones
    
    #ASSIGN RATING FAIRNESS AND WEIGHTED RATING FAIRNESS AS 1 TO DUMMY RATINGS
    ratingFairnessList = ones #important
    weightedRatingFairnesList = ones 
    #ASSIGN (score-goodness) as None TO DUMMY RATINGS SINCE THIS WILL BE CALCULATED IN REV2   
    score_goodness_difference_metricList = [None]*numberOfRatings #will be computed in rev2
   
    #EDGE CASE : IF THIS NOTE DOES NOT HAVE ANY RATINGS, ITS FINE SICNE WE ARE ONLY ADDING RATING ROWS HERE
    newDummyRowsDf = pd.DataFrame(data=zip(noteIdTempList,participantIdTempList,helpfulTempList,notHelpfulTempList,goodnessList,ratingsList,score_goodness_difference_metricList,ratingFairnessList,weightedRatingFairnesList,goodness_term1List), columns=list(ratingsOld.columns))
    ratingsNew = ratingsOld.append(newDummyRowsDf, ignore_index=True)
    return ratingsNew

def rev2(ratings,notes,init_lambda,init_goodness,convergence_threshold,alpha1,beta1,gamma1,delta1,mu_r,mu_w,mu_t,mu_g):
    
    lambda1 = init_lambda
    lambda2 = init_lambda
    lambda3 = init_lambda

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

def findNumberOfAccountsNeeded(tweetId,ratingsBase,notesBase,
                               candidateNotesIds,currentlyTopGoodNotesIds,
                               init_lambda=None,
                               init_goodness=None,
                               convergence_threshold=None,
                               alpha1=None,beta1=None,gamma1=None,delta1=None,
                               mu_r=None,mu_w=None,mu_t=None,mu_g=None,
                               maxCurrentTopGoodNotes=None,
                               minRatingsNeeded = None,
                               minGoodnessNeeded = None,
                               maxAccountsToBreak = None,
                               isInsertion=None,isReplacement=None):
    
    #For a random note, do a run through of all the possible number of accounts one can use 
    #to rate this note helpful and bring it to "Currently Rated Helpful" (if it is currently not)
      
    randomNoteId = random.choice(list(candidateNotesIds))
    numberOfAccounts = 1 #number of required ratings - no ratings it already has
    while(True):
  
        #Create as many dummy participant IDs as the number of accounts
        dummyParticipantIdList = [createDummyParticipantId(32,ratingsBase) for i in range(0,numberOfAccounts)]
        
        if isInsertion:
            #Each of these participantIds adds a dummy ratings for each note(HELPFUL=1)
            ratingsOld = ratingsBase.copy(deep=True)
            ratingsNew = addDummyRatings(randomNoteId,dummyParticipantIdList,ratingsOld,notesBase,helpful=1)
        
        if isReplacement:
            #At the same time add NOT HELPFUL to the top 3 notes
            for top3NoteId in currentlyTopGoodNotesIds:
                ratingsNew = addDummyRatings(top3NoteId,dummyParticipantIdList,ratingsNew,notesBase,helpful=0)
        
        #PERFORM REV2 AGAIN NOW THAT WE HAVE DUMMY RATINGS
        notesRev2,ratingsRev2 = rev2(ratingsNew,notesBase,init_lambda,init_goodness,convergence_threshold,alpha1,beta1,gamma1,delta1,mu_r,mu_w,mu_t,mu_g)
    
        #Get the top good notes again!
        currentTopGoodNotesNew = getCurrentTopGoodNotesForTweet(notesRev2,ratingsRev2,tweetId,topNotes = maxCurrentTopGoodNotes,minRatingsNeeded = minRatingsNeeded,minGoodnessNeeded = minGoodnessNeeded)
        currentTopGoodNotesIdsNew = set(currentTopGoodNotesNew['noteId'])

        #Does our (current note) occur in the currentlyTopGoodNotesIdsNew?
        if randomNoteId in currentTopGoodNotesIdsNew:
            return numberOfAccounts
        
        #If not, do this for +1 more accounts
        numberOfAccounts += 1 
        #If the note doesnt come in top good even after 10 notes
        if numberOfAccounts == maxAccountsToBreak:
            return maxAccountsToBreak
        
def processTweet(results,tweetId,keywordArgs):
    
    ratingsBaseGlobal = keywordArgs['ratingsBaseGlobal']
    notesBaseGlobal = keywordArgs['notesBaseGlobal']
    maxCurrentTopGoodNotes = keywordArgs['maxCurrentTopGoodNotes']
    minRatingsNeeded = keywordArgs['minRatingsNeeded']
    maxAccountsToBreak = keywordArgs['maxAccountsToBreak']
    minGoodnessNeeded = keywordArgs['minGoodnessNeeded']
    init_lambda = keywordArgs['init_lambda']
    init_goodness = keywordArgs['init_goodness']
    alpha1 = keywordArgs['alpha1']
    beta1 = keywordArgs['beta1']
    gamma1 = keywordArgs['gamma1']
    delta1 = keywordArgs['delta1']
    mu_r = keywordArgs['mu_r']
    mu_w = keywordArgs['mu_w']
    mu_t = keywordArgs['mu_t']
    mu_g = keywordArgs['mu_g']
    convergence_threshold = keywordArgs['convergence_threshold']
    
    #Get all notes for this tweet
    notesForTweet = notesBaseGlobal[notesBaseGlobal['tweetId']==tweetId]
    allNotesSet = set(notesForTweet['noteId'])
     
    #Get current top good notes for the tweet using the hyperparamters
    currentTopGoodNotes = getCurrentTopGoodNotesForTweet(notesBaseGlobal,ratingsBaseGlobal,
                                                         tweetId,topNotes = maxCurrentTopGoodNotes,
                                                         minRatingsNeeded = minRatingsNeeded,
                                                         minGoodnessNeeded = minGoodnessNeeded)
    currentlyTopGoodNotesIds = set(currentTopGoodNotes['noteId'])
    
    #Candidate Notes
    candidateNotesIds = allNotesSet - currentlyTopGoodNotesIds

    limit = maxCurrentTopGoodNotes
    if len(candidateNotesIds)==0:
        results["default"].append(tweetId)
        
    elif len(candidateNotesIds)>0 and len(currentlyTopGoodNotesIds)<limit:
        results["insertion"][tweetId] = findNumberOfAccountsNeeded(
                            tweetId,ratingsBaseGlobal,notesBaseGlobal,
                            candidateNotesIds,currentlyTopGoodNotesIds,
                            init_lambda=init_lambda,
                            init_goodness=init_goodness,
                            convergence_threshold=convergence_threshold,
                            alpha1=alpha1,beta1=beta1,gamma1=gamma1,delta1=delta1,
                            mu_r=mu_r,mu_w=mu_w,mu_t=mu_t,mu_g=mu_g,
                            maxCurrentTopGoodNotes=maxCurrentTopGoodNotes,
                            minRatingsNeeded = minRatingsNeeded,
                            minGoodnessNeeded = minGoodnessNeeded,
                            maxAccountsToBreak = maxAccountsToBreak,
                            isInsertion=True,isReplacement=False)

    elif len(allNotesSet)>limit and len(currentlyTopGoodNotesIds)==limit:
        results["replacement"][tweetId] = findNumberOfAccountsNeeded(
                                 tweetId,ratingsBaseGlobal,notesBaseGlobal,
                                 candidateNotesIds,currentlyTopGoodNotesIds,
                               init_lambda=init_lambda,
                               init_goodness=init_goodness,
                               convergence_threshold=convergence_threshold,
                               alpha1=alpha1,beta1=beta1,gamma1=gamma1,delta1=delta1,
                               mu_r=mu_r,mu_w=mu_w,mu_t=mu_t,mu_g=mu_g,
                               maxCurrentTopGoodNotes=maxCurrentTopGoodNotes,
                               minRatingsNeeded = minRatingsNeeded,
                               minGoodnessNeeded = minGoodnessNeeded,
                               maxAccountsToBreak = maxAccountsToBreak,
                               isInsertion=True,isReplacement=True)
    else:
        print("Case not allotted ? : ",tweetId)
        
    return results
    
##########################################################################################
    
if __name__ == '__main__':
    num_processes = 20
    maxCurrentTopGoodNotes = 1
    minRatingsNeeded = 5
    maxAccountsToBreak = 10
    #CHANGE GOODNESS AS NEEDED
    minGoodnessNeeded = 0.01
    init_lambda = 0.1
    init_goodness = 1
    convergence_threshold = 0.001
    
    alpha1 = 1
    beta1 = 1
    gamma1 = 1
    delta1 = 1
    
    notesGlobal = pd.read_csv("notes-00000-13-04-21.tsv", sep='\t')
    ratingsGlobal = pd.read_csv("ratings-00000-13-04-21.tsv", sep='\t')
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
    
    #PERFORM REV2, THIS WILL BE THE INITIAL STATE
    print("Starting initial REV2....")
    notesBaseGlobal,ratingsBaseGlobal = rev2(ratingsGlobal,notesGlobal,init_lambda,init_goodness,convergence_threshold,alpha1,beta1,gamma1,delta1,mu_r,mu_w,mu_t,mu_g)
    print("Initial REV2 done!")
    
    totalTweets = list(set(notesGlobal['tweetId']))
    #CHANGE THE CHUNK OF TWEETS YOU WANT TO PROCESS 
    #IF YOU WANT TO DO IT IN BATCHES 
    #start,end = 0,1225
    #start,end = 1225,2450
    #start,end = 2450,3675
    start,end = 3675,4900
    allTweets = totalTweets[start:end]
    keywordArgs = {'ratingsBaseGlobal':ratingsBaseGlobal,
                    'notesBaseGlobal':notesBaseGlobal,
                    'maxCurrentTopGoodNotes' : maxCurrentTopGoodNotes,
                    'minRatingsNeeded' : minRatingsNeeded,
                    'maxAccountsToBreak' : maxAccountsToBreak,
                    'minGoodnessNeeded' : minGoodnessNeeded,
                    'init_lambda' : init_lambda,
                    'init_goodness' : init_goodness,
                    'convergence_threshold' : convergence_threshold,
                    'alpha1' : alpha1,
                    'beta1' : beta1,
                    'gamma1' : gamma1,
                    'delta1' : delta1,
                    'mu_r' : mu_r,
                    'mu_w' : mu_w,
                    'mu_t' : mu_t,
                    'mu_g' : mu_g}
    
    results = {}
    for tweetId in allTweets:
        results = processTweet(results,tweetId,keywordArgs)
        
    insertion = dict(results['insertion'])
    replacement = dict(results['replacement'])
    default = list(results['default'])
    with open('results/rev2-cold-001-insertion'+str(start)+'_to_'+str(end)+'.pickle', 'wb') as handle:
        pickle.dump(insertion, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('results/rev2-cold-001-replacement'+str(start)+'_to_'+str(end)+'.pickle', 'wb') as handle:
        pickle.dump(replacement, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('results/rev2-cold-001-default'+str(start)+'_to_'+str(end)+'.pickle', 'wb') as handle:
        pickle.dump(default, handle, protocol=pickle.HIGHEST_PROTOCOL)    
