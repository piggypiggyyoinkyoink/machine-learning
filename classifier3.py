import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score


class SmoothedClassConditionalVotingClassifier2(BaseEstimator, ClassifierMixin):
    """
    A class-conditional weighted voting classifier.
    
    For each estimator j and class c, compute:
    
        weight[j, c] = smoothed F1 score of estimator j on class c
    
    where smoothing is:
    
        smoothed_F1 = (F1 + epsilon) / (1 + epsilon)
    
    At prediction time:
      - If voting="hard": votes are weighted by class-conditional weights
      - If voting="soft": predict_proba is weighted per-class
    """

    def __init__(self, estimators, voting="hard", epsilon=1.0):
        """
        Parameters
        ----------
        estimators : list of (str, estimator)
            List of named base estimators.
        voting : {"hard", "soft"}
            Voting strategy.
        epsilon : float
            Laplace smoothing parameter.
        """
        self.estimators = estimators
        self.voting = voting
        self.epsilon = epsilon
        self.classes_ = None
        self.class_weights_ = None
        self.estimators_ = None

    # ----------------------------------------------------------
    def fit(self, X, y):
        """
        Train estimators on DSEL (the ensemble validation set)
        and compute class-conditional smoothed F1 weights.
        """
        # Fit each estimator
        self.estimators_ = [est
                            for name, est in self.estimators]
        
        self.classes_ = np.unique(y)
        C = len(self.classes_)
        M = len(self.estimators_)
        weights = np.zeros((M, C), dtype=float)

        # Compute smoothed F1 per class per estimator
        for j, (clf) in enumerate(self.estimators_):
            y_pred = clf.predict(X)

            for i, cls in enumerate(self.classes_):
                # F1 for this class
                f1 = f1_score(y == cls, y_pred == cls)

                # Laplace smoothing to stabilize small classes
                smoothed = (f1 + self.epsilon) / (1.0 + self.epsilon)

                weights[j, i] = smoothed

        self.class_weights_ = weights  # shape (M, C)
        return self

    # ----------------------------------------------------------
    def predict_proba(self, X):
        """
        Weighted-probability voting (class-conditional).
        """
        if self.voting != "soft":
            raise ValueError("predict_proba is only valid for voting='soft'")

        M = len(self.estimators_)
        C = len(self.classes_)
        proba_sum = np.zeros((X.shape[0], C), dtype=float)

        for j, (clf) in enumerate(self.estimators_):
            p = clf.predict_proba(X)  # shape (N, C)

            # Multiply each class column by its weight
            weighted_p = p * self.class_weights_[j]

            proba_sum += weighted_p

        # Normalize row-wise
        proba_sum /= proba_sum.sum(axis=1, keepdims=True)
        return proba_sum

    # ----------------------------------------------------------
    def predict(self, X):
        """
        Weighted hard or soft voting.
        """
        if self.voting == "soft":
            # pick argmax of weighted probabilities
            proba = self.predict_proba(X)
            idx = np.argmax(proba, axis=1)
            return self.classes_[idx]

        # ------------ HARD VOTING --------------------
        M, C = self.class_weights_.shape
        N = X.shape[0]
        vote_scores = np.zeros((N, C), dtype=float)

        for j, (clf) in enumerate(self.estimators_):
            preds = clf.predict(X)
            for i, cls in enumerate(self.classes_):
                # Add weight if clf predicts class cls
                vote_scores[:, i] += (preds == cls) * self.class_weights_[j, i]

        # final decision
        idx = np.argmax(vote_scores, axis=1)
        return self.classes_[idx]

    def diagnostics(self):
        """
        Print useful debugging information about per-class weights
        and per-estimator specialization.
        """
        if self.class_weights_ is None:
            print("Model not fitted yet.")
            return

        weights = self.class_weights_
        M, C = weights.shape

        print("\n" + "="*70)
        print("               CLASS-CONDITIONAL WEIGHT DIAGNOSTICS")
        print("="*70)

        # --- Global stats ---
        print("\nGLOBAL WEIGHT STATS:")
        print(f"  Mean weight:  {weights.mean():.4f}")
        print(f"  Std:          {weights.std():.4f}")
        print(f"  Min weight:   {weights.min():.4f}")
        print(f"  Max weight:   {weights.max():.4f}")

        # --- Per class summary ---
        print("\nPER-CLASS WEIGHT STATS:")
        for i, cls in enumerate(self.classes_):
            cls_w = weights[:, i]
            print(f"  Class {cls:>3}: mean={cls_w.mean():.4f}, "
                  f"std={cls_w.std():.4f}, min={cls_w.min():.4f}, max={cls_w.max():.4f}")

        # --- Per estimator ---
        print("\nPER-ESTIMATOR SPECIALIZATION:")
        for j, (name, _) in enumerate(self.estimators):
            w = weights[j]
            entropy = -np.sum((w / w.sum()) * np.log((w / w.sum()) + 1e-12))
            print(f"  {name:15s}: mean={w.mean():.4f}, std={w.std():.4f}, "
                  f"min={w.min():.4f}, max={w.max():.4f}, entropy={entropy:.4f}")

        # --- Identify unstable weights (likely due to low support classes) ---
        print("\nUNSTABLE OR LOW-SUPPORT CLASS CHECK:")
        global_mean = weights.mean(axis=0)

        for i, cls in enumerate(self.classes_):
            if global_mean[i] < 0.2:  # threshold for suspiciously low competence
                print(f"  Class {cls} has low average weight ({global_mean[i]:.3f}). "
                      f"Possible small sample or poor consensus.")

        print("="*70 + "\n")