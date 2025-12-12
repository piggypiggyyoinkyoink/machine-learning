import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.metrics import f1_score, accuracy_score


class ClassWeightedVotingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimators, alpha:float=1.0, weighted = True):
        self.estimators = estimators
        self._estimator_type = "classifier"
        self.alpha = alpha
        self.weighted = weighted

        #set once fitted:
        self.classes = None
        self.weights = None  

    


    def fit(self, X, y):
        # Check that X and y have correct shape, set n_features_in_, etc.
        X, y = validate_data(self, X, y)
        # Store the classes seen during fit
        self.classes = unique_labels(y)

        num_classes = len(self.classes)
        num_estimators = len(self.estimators)

        #initialise array for weights 
        weights = np.zeros((num_estimators, num_classes), dtype=float)

        if self.weighted:
            accs = np.zeros(num_estimators, dtype=float)

            for j in range(len(self.estimators)):
                est = self.estimators[j][1]
                try:
                    probability_matrix = est.predict_proba(X)  #shape: (num_samples, num_classes)
                except:
                    #no predict_proba method
                    raise Exception("Estimator does not have predict_proba method")
                #get probability matrix of all classes for all samples for this estimator

                #Get the predicted class for every sample for this estimator
                #(this is the class with the highest probability for that sample in the probability matrix)
                y_pred = self.classes[np.argmax(probability_matrix, axis=1)]

                #use f1 score for each class to compute weight for that class for this estimator.
                for i in range(len(self.classes)):
                    f1 = f1_score(y == self.classes[i], y_pred == self.classes[i])
                    smoothed = (f1 + self.alpha) / (1.0 + self.alpha)
                    weights[j, i] = smoothed

                #get overall accuracy score for each estimator against X and y
                accs[j] = accuracy_score(y, y_pred)

            #normalise accuracy scores to [0,1]
            accs = (accs - accs.min()) / (accs.max() - accs.min() + 1e-12) #add small number so weights never zero
            weights = weights * accs[:, None] #multiply weights by overall accuracy of the estimator

            #normalise weights per estimator
            row_sums = weights.sum(axis=1, keepdims=True) + 1e-12 #to avoid division by zero
            weights = weights / row_sums
        else:
            #if not weighted, all weights are 1
            weights.fill(1.0)
        self.weights = weights #shape: (num_estimators, num_classes))
        return self




    #predict_proba method needed so i can use a CWVClassifier as an estimator for another CWVClassifier
    def predict_proba(self, X):
        if self.weights is None:
            raise Exception("Model not fitted yet")

        X = np.array(X)
        #initialise probability sum array
        prob_sum = np.zeros((X.shape[0], len(self.classes)), dtype=float)

        for j in range(len(self.estimators)):
            est = self.estimators[j][1]
            try:
                #get probability matrix for estimator
                probability_matrix = est.predict_proba(X)
            except:
                #no predict_proba method
                raise Exception("Estimator does not have predict_proba method")
            #multiply by weight then add to the sum
            prob_sum += probability_matrix * self.weights[j][None, :]

        #normalise rows to be probabilities
        prob_sum = prob_sum / (prob_sum.sum(axis=1, keepdims=True) + 1e-12)
        return prob_sum
    



    def predict(self, X):
        #predict provbabilities of each class
        probabilities = self.predict_proba(X)
        #return class of highest probability
        idx = np.argmax(probabilities, axis=1)
        return self.classes[idx]

    