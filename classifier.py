import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors


class LocalAccuracyVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    A VotingClassifier variant that computes classifier weights per sample
    based on local accuracy in the neighborhood of the sample.

    Parameters
    ----------
    estimators : list of (str, estimator)
        List of name/estimator pairs.
    k : int, default=15
        Number of neighbors used to compute local accuracy.
    voting : {'hard', 'soft'}, default='hard'
        Voting strategy.
    """

    def __init__(self, estimators, k=15, voting="soft"):
        self.estimators = estimators
        self.k = k
        self.voting = voting

    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # Check that X and y have correct shape, set n_features_in_, etc.
        X, y = validate_data(self, X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        #fit nearest neighbor model
        self.nn_ = NearestNeighbors(n_neighbors=self.k)
        self.nn_.fit(X)

        #fit all estimators
        self.estimators_ = []
        for name, est in self.estimators:
        
            self.estimators_.append((name, est))

        return self

    def _local_weights(self, x):

        #find k nearest neighbors
        _, idx = self.nn_.kneighbors([x], return_distance=True)
        idx = idx[0]
        #print(idx)
        y_local = self.y_[idx]

        weights = []
        for _, est in self.estimators_:
            pred_local = est.predict(self.X_[idx])
            acc = np.mean(pred_local == y_local)
            #if acc != 1.0:
                #print("ACC:", acc)
            #set weight to very small value so it isnt 0.
            acc = max(acc, 1e-6)
            weights.append(acc)

        weights = np.array(weights)
        #print("WEIGHTS:", weights)
        #print("RETURNED:", weights / weights.sum())
        return weights / weights.sum()

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = validate_data(self, X, reset=False)

        X = np.asarray(X)
        preds = []

        for x in X:
            w = self._local_weights(x)

            # Hard voting: weighted class-frequency
            if self.voting == "hard":
                scores = {cls: 0.0 for cls in self.classes_}
                for weight, (_, est) in zip(w, self.estimators_):
                    pred = est.predict([x])[0]
                    scores[pred] += weight
                preds.append(max(scores, key=scores.get))

            # Soft voting: weighted soft probabilities
            else:
                #get array of predictons from each estimator
                probas = np.array([
                    est.predict_proba([x])[0]
                    for _, est in self.estimators_
                ])
                #weight the predictions according to nn performance
                weighted = probas * w[:, None]
                avg = weighted.sum(axis=0)#add up predicted probabilities for each class across the estimators
                pred_class = self.classes_[np.argmax(avg)] #class with highest overall probability
                preds.append(pred_class)

        return np.array(preds)

    
    def predict_proba(self, X):
        if self.voting != "soft":
            raise AttributeError("predict_proba only available when voting='soft'")

        X = np.asarray(X)
        probs = []

        for x in X:
            w = self._local_weights(x)
            probas = np.array([
                est.predict_proba([x])[0]
                for _, est in self.estimators_
            ])
            weighted = probas * w[:, None]
            probs.append(weighted.sum(axis=0))

        return np.array(probs)
