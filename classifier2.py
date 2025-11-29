import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import precision_recall_fscore_support


class ClassConditionalVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    A custom VotingClassifier that assigns per-class weights to each estimator
    based on per-class F1 scores measured on an ensemble-validation set.

    Parameters
    ----------
    estimators : list
        List of base estimators (unfitted).
    
    normalize : bool (default=True)
        If True, weights for each class are normalized across estimators.
    """

    def __init__(self, estimators, normalize=True):
        self.estimators = estimators
        self.normalize = normalize

    # ---------------------------------------------------------
    # Fit models using the *training* set, then compute
    # class-conditional F1 weights using the *validation* set.
    # ---------------------------------------------------------
    def fit(self, X_train, y_train):
        self.classes_ = np.unique(y_train)
        self.n_classes_ = len(self.classes_)
        self.n_estimators_ = len(self.estimators)

        # 
        self.models_ = [est
                        for i, est in self.estimators]

        # Storage for class-conditional weights:
        # shape = (n_estimators, n_classes)
        self.class_weights_ = np.zeros((self.n_estimators_, self.n_classes_))

        # Compute predictions on validation set
        val_preds = np.column_stack([m.predict(X_train) for m in self.models_])

        # Compute per-class precision/recall/F1 for each estimator
        for j in range(self.n_estimators_):
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_train, 
                val_preds[:, j],
                labels=self.classes_,
                zero_division=0
            )
            self.class_weights_[j, :] = f1

        # Optionally normalize weights per class
        if self.normalize:
            for c in range(self.n_classes_):
                total = self.class_weights_[:, c].sum()
                if total > 0:
                    self.class_weights_[:, c] /= total

        return self

    # ---------------------------------------------------------
    # Weighted class-conditional voting
    # ---------------------------------------------------------
    def predict(self, X):
        # Predictions for each estimator
        preds = np.column_stack([m.predict(X) for m in self.models_])

        # Vote matrix: shape = (n_samples, n_classes)
        votes = np.zeros((X.shape[0], self.n_classes_))

        # Add weights for each estimator's predicted class
        for j in range(self.n_estimators_):
            for c_index, c in enumerate(self.classes_):
                mask = (preds[:, j] == c)
                votes[mask, c_index] += self.class_weights_[j, c_index]

        winner_indices = np.argmax(votes, axis=1)
        return self.classes_[winner_indices]

    # ---------------------------------------------------------
    # Optional: probability-like soft output
    # ---------------------------------------------------------
    def predict_proba(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models_])

        votes = np.zeros((X.shape[0], self.n_classes_))
        for j in range(self.n_estimators_):
            for c_index, c in enumerate(self.classes_):
                votes[preds[:, j] == c, c_index] += self.class_weights_[j, c_index]

        # Normalize row-wise
        row_sums = votes.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-12
        return votes / row_sums
