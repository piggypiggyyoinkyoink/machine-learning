import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
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

    def __init__(self, estimators, k=15, voting="hard"):
        self.estimators = estimators
        self.k = k
        self.voting = voting

    # -------------------------
    # Fit
    # -------------------------
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # Store training data
        self.X_ = X
        self.y_ = y

        # Store class labels (IMPORTANT: for correct mapping)
        self.classes_ = np.unique(y)

        # Fit nearest neighbor model
        self.nn_ = NearestNeighbors(n_neighbors=self.k)
        self.nn_.fit(X)

        # Fit all estimators
        self.estimators_ = []
        for name, est in self.estimators:
            cloned = clone(est)
            cloned.fit(X, y)
            self.estimators_.append((name, cloned))

        return self

    # -------------------------
    # Compute dynamic weights
    # -------------------------
    def _local_weights(self, x):
        """Compute per-estimator accuracy in local neighborhood of x."""

        # Find k nearest neighbors
        _, idx = self.nn_.kneighbors([x], return_distance=True)
        idx = idx[0]
        y_local = self.y_[idx]

        weights = []
        for _, est in self.estimators_:
            pred_local = est.predict(self.X_[idx])
            acc = np.mean(pred_local == y_local)
            acc = max(acc, 1e-6)      # avoid zero weight
            weights.append(acc)

        weights = np.array(weights)
        return weights / weights.sum()

    # -------------------------
    # Predict (hard + soft)
    # -------------------------
    def predict(self, X):
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
            elif self.voting == "soft":
                probas = np.array([
                    est.predict_proba([x])[0]
                    for _, est in self.estimators_
                ])
                weighted = probas * w[:, None]
                avg = weighted.sum(axis=0)
                pred_class = self.classes_[np.argmax(avg)]
                preds.append(pred_class)

            else:
                raise ValueError("voting must be 'hard' or 'soft'")

        return np.array(preds)

    # -------------------------
    # predict_proba (soft only)
    # -------------------------
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
