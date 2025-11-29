import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


class ClassConditionalAccuracyVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    Voting classifier that uses class-conditional accuracy for each estimator,
    smoothed toward 0.5 using a mixing factor alpha.

    Parameters
    ----------
    estimators : list of (str, estimator)
        List of named sklearn estimators.

    alpha : float (0–1)
        Smoothing factor. 1.0 = no smoothing. 0.0 = all weights = 0.5.
        Recommended: 0.7–0.9 for sparse DSEL.

    Attributes
    ----------
    class_cond_acc_ : array of shape (n_estimators, n_classes)
        Smoothed class-conditional accuracies.

    classes_ : array of shape (n_classes,)
        The unique class labels.

    diagnostics_ : dict
        Diagnostic statistics for weight stability.
    """

    def __init__(self, estimators, alpha=0.8):
        self.estimators = estimators
        self.alpha = alpha

    # ----------------------------------------------------------
    # --------------------- FIT METHOD -------------------------
    # ----------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_estimators = len(self.estimators)

        # DO NOT clone or refit
        # We simply store the pre-trained models.
        self.estimators_ = [(name, est) for name, est in self.estimators]

        # Compute class-conditional accuracies
        class_acc = np.zeros((n_estimators, n_classes))

        for j, (name, est) in enumerate(self.estimators_):
            preds = est.predict(X)

            for c in range(n_classes):
                mask = (y == c)
                class_acc[j, c] = np.mean(preds[mask] == c)

        # Apply smoothing
        self.class_cond_acc_ = (
            self.alpha * class_acc + (1 - self.alpha) * 0.5
        )

        self._compute_diagnostics(class_acc)
        return self

    # ----------------------------------------------------------
    # --------------------- PREDICT ----------------------------
    # ----------------------------------------------------------

    def predict(self, X):
        check_is_fitted(self, "class_cond_acc_")
        X = np.asarray(X)

        n_samples = X.shape[0]
        n_estimators, n_classes = self.class_cond_acc_.shape

        # Each estimator's prediction
        preds = np.column_stack([
            est.predict(X) for _, est in self.estimators_
        ])

        # Weighted voting: score[c] = sum_j (1_{pred_j=c} * weight_j,c)
        scores = np.zeros((n_samples, n_classes))

        for j in range(n_estimators):
            for c in range(n_classes):
                mask = preds[:, j] == c
                scores[mask, c] += self.class_cond_acc_[j, c]

        return np.argmax(scores, axis=1)

    # ----------------------------------------------------------
    # -------------------- PREDICT PROBA -----------------------
    # ----------------------------------------------------------

    def predict_proba(self, X):
        check_is_fitted(self, "class_cond_acc_")
        X = np.asarray(X)

        n_samples = X.shape[0]
        n_estimators, n_classes = self.class_cond_acc_.shape

        preds = np.column_stack([
            est.predict(X) for _, est in self.estimators_
        ])

        scores = np.zeros((n_samples, n_classes))

        for j in range(n_estimators):
            for c in range(n_classes):
                mask = preds[:, j] == c
                scores[mask, c] += self.class_cond_acc_[j, c]

        # Normalize row-wise to make probabilities
        probs = scores / np.sum(scores, axis=1, keepdims=True)
        return probs

    def _compute_diagnostics(self, raw_acc):
        """
        Stores diagnostics in self.diagnostics_:
        - raw accuracy mean/std
        - smoothed accuracy mean/std
        - per-class variance
        - per-class entropy (instability indicator)
        """
        smoothed = self.class_cond_acc_
        n_estimators, n_classes = smoothed.shape

        per_class_var = np.var(smoothed, axis=0)
        per_class_mean = np.mean(smoothed, axis=0)

        # Entropy across models for each class
        # Normalize weights per class → interpret as pseudo-probabilities
        eps = 1e-12
        norm = smoothed / np.sum(smoothed, axis=0, keepdims=True)
        entropy = -np.sum(norm * np.log(norm + eps), axis=0)

        self.diagnostics_ = {
            "raw_accuracy_mean": float(np.mean(raw_acc)),
            "raw_accuracy_std": float(np.std(raw_acc)),
            "smoothed_mean": float(np.mean(smoothed)),
            "smoothed_std": float(np.std(smoothed)),
            "per_class_mean": per_class_mean,
            "per_class_variance": per_class_var,
            "entropy_per_class": entropy,
            "max_entropy": float(np.max(entropy)),
            "min_entropy": float(np.min(entropy)),
        }