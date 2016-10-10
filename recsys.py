""" Example PySpark ALS application
"""
from IPython.core.display import display
import numpy as np

from pyspark import SparkContext  # pylint: disable=import-error
from pyspark.mllib.recommendation import ALS, Rating, MatrixFactorizationModel  # pylint: disable=import-error

import math

def parse_rating(line):
    """ Parse Movielens Rating line to Rating object.
        UserID::MovieID::Rating::Timestamp
    """
    line = line.split('::')
    #print line
    #print line[0] #, line[1], line[2]
    #print int(line[0]), int(line[1]), int(line[2])
    return Rating(int(line[0]), int(line[1]), float(line[2]))
    #return Rating(int(line[0]), int(line[1]), int(line[2]))


def parse_movie(line):
    """ Parse Movielens Movie line to Movie tuple.
        MovieID::Title::Genres
    """
    line = line.split('::')
    return (line[0], line[1])


def computeError(predictedRDD, actualRDD):
    """ Compute the root mean squared error between predicted and actual
    Args:
        predictedRDD: predicted ratings for each movie and each user where each entry is in the form
                      (UserID, MovieID, Rating)
        actualRDD: actual ratings where each entry is in the form (UserID, MovieID, Rating)
    Returns:
        RSME (float): computed RSME value
    """
    # Transform predictedRDD into the tuples of the form ((UserID, MovieID), Rating)
    predictedReformattedRDD = predictedRDD.map(lambda (UserID, MovieID, Rating): ((UserID, MovieID), Rating))

    # Transform actualRDD into the tuples of the form ((UserID, MovieID), Rating)
    actualReformattedRDD = actualRDD.map(lambda (UserID, MovieID, Rating): ((UserID, MovieID), Rating))

    # Compute the squared error for each matching entry (i.e., the same (User ID, Movie ID) in each
    # RDD) in the reformatted RDDs using RDD transformtions - do not use collect()
    squaredErrorsRDD = (predictedReformattedRDD
                        .join(actualReformattedRDD)
                       .map(lambda ((UserID, MovieID), (PredRating, ActRating)): (PredRating - ActRating)**2))

    
    # Compute the total squared error - do not use collect()
    totalError = squaredErrorsRDD.reduce(lambda a,b: a+b)

    # Count the number of entries for which you computed the total squared error
    numRatings = squaredErrorsRDD.count()

    # Using the total squared error and the number of entries, compute the RSME
    return math.sqrt(1.0*totalError/numRatings)





def main():
    """ Train and evaluate an ALS recommender.
    """
    # Set up environment
    sc = SparkContext("local[*]", "RecSys")

    # Load and parse the data
    #data = sc.textFile("./data/ratings.dat") 
    ratingsRDD = sc.textFile("file:///Users/xicheng/Dropbox/Crackcode/BitTiger/0603movie/ml-10M100K/ratings.dat")
    
    moviesRDD = sc.textFile("file:///Users/xicheng/Dropbox/Crackcode/BitTiger/0603movie/ml-10M100K/movies.dat")\
                   .map(parse_movie)

    trainingRDD, validationRDD, testRDD = ratingsRDD.randomSplit([6, 2, 2], seed=0L)
    print 'Training: %s, validation: %s, test: %s\n' % (trainingRDD.count(),
                                                        validationRDD.count(),
                                                        testRDD.count())

    #ratings = data.map(lambda l: l.split()).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()

    #data = sc.textFile("./ratings.dat")
    ratingsTrain = trainingRDD.map(parse_rating).cache()

    #movies = sc.textFile("./data/movies.dat")\
    #           .map(parse_movie)



    # Evaluate the model on training data
    predictdata = validationRDD.map(lambda p: (p[0], p[1]))

    #colNames = ["userID", "movieID"]
    #df = data.toDF(colNames)
    


    # Build the recommendation model using Alternating Least Squares
    rank = 10
    iterations = 10 #20

    theLambda = 0.01   #  use cross-validation to adjust

    bestMSE = 100
    bestRank = 100

    for i in range(10,20):
        rank = i
        #model = ALS.train(ratings, rank, iterations) 
        model = ALS.train(ratingsTrain, rank, iterations, theLambda) 

        predictions = model.predictAll(predictdata)\
                       .map(lambda r: ((r[0], r[1]), r[2]))

        rates_and_preds = ratingsTrain.map(lambda r: ((r[0], r[1]), r[2]))\
                             .join(predictions)

        MSE = rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean()

        print("Mean Squared Error = " + str(MSE))
        print(rank ,"mse = {:.3f}".format(MSE))
        if MSE < bestMSE :
            bestMSE = MSE
            bestRank = i

    print (bestMSE, bestRank)



if __name__ == "__main__":
    main()