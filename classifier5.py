import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted

def _softmax(x, axis=1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

class SmoothedClassConditionalVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    Improved Smoothed Class-Conditional Voting Classifier.

    Parameters
    ----------
    estimators : list of (name, estimator)
        Base estimators. They can be already-fitted or unfitted.
    voting : {"soft","hard"}
        Voting strategy. "soft" uses predict_proba (preferred). "hard" uses weighted votes.
    epsilon : float
        Laplace smoothing parameter applied to per-class F1s.
    weight_by_accuracy : bool
        Multiply per-class weights by the estimator's global accuracy (stabilizes weak models).
    calibrate : bool
        If True, fit a CalibratedClassifierCV on each (cloned) estimator using DSEL in fit().
        If provided estimators are already fitted and calibrate=True, they will be cloned and calibrated.
    verbose : int
        Print diagnostics in fit() if >0.
    """

    def __init__(self, estimators, voting="soft", epsilon=1.0, weight_by_accuracy=False, calibrate=False, verbose=0):
        self.estimators = estimators
        self._estimator_type = "classifier"
        self.voting = voting
        self.epsilon = float(epsilon)
        self.weight_by_accuracy = bool(weight_by_accuracy)
        self.calibrate = bool(calibrate)
        self.verbose = int(verbose)

        # filled at fit()
        self.estimators_ = None
        self.classes_ = None
        self.class_weights_ = None  # shape (M, C): per-estimator × per-class weight
        self.global_acc_ = None

    # -------------------------
    def _ensure_estimator(self, est):
        """Return estimator (clone if unfitted and we need to fit it)."""
        return est

    # -------------------------
    def fit(self, X, y):
        """
        Fit (or calibrate) base estimators on DSEL (ensemble validation set) and compute class-conditional weights.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            DSEL data used to evaluate and optionally calibrate base classifiers.
        y : array-like, shape (n_samples,)
            True labels for DSEL.
        """
        # Establish global class list
        self.classes_ = np.unique(y)
        C = len(self.classes_)
        M = len(self.estimators)

        # Prepare storage
        self.estimators_ = []
        weights = np.zeros((M, C), dtype=float)
        accs = np.zeros(M, dtype=float)

        # Fit or calibrate each estimator on DSEL if calibrate==True,
        # otherwise assume estimator already trained. We do not re-train base estimators here.
        for j, (name, est) in enumerate(self.estimators):
            # If calibrate: clone and fit CalibratedClassifierCV on DSEL
            if self.calibrate:
                # clone + calibrate on DSEL (usually a small CV inside)
                calibrated = CalibratedClassifierCV(clone(est), cv="prefit" if hasattr(est, "predict") else 3)
                # If estimator is already fitted, prefer prefitting path:
                try:
                    # attempt to fit calibrated wrapper; if est is not fitted, it will be fit inside CV
                    calibrated.fit(X, y)
                    est_j = calibrated
                except Exception:
                    # fallback: clone and fit directly (non-prefit)
                    c2 = clone(est)
                    c2.fit(X, y)
                    calibrated = CalibratedClassifierCV(c2, cv=3)
                    calibrated.fit(X, y)
                    est_j = calibrated
            else:
                # use provided estimator as-is (assume pre-fitted)
                est_j = est

            self.estimators_.append(est_j)

            # produce predictions / probabilities on DSEL
            # handle classifiers without predict_proba
            if hasattr(est_j, "predict_proba"):
                p = est_j.predict_proba(X)  # shape (n_samples, n_est_classes)
            elif hasattr(est_j, "decision_function"):
                df = est_j.decision_function(X)
                # decision_function may be shape (n_samples,) for binary; convert carefully
                if df.ndim == 1:
                    df = np.vstack([-df, df]).T
                p = _softmax(df, axis=1)
            else:
                # last resort: use hard predictions as degenerate probabilities
                preds = est_j.predict(X)
                p = np.zeros((X.shape[0], len(np.unique(preds))), dtype=float)
                # Map to columns
                est_classes = np.unique(preds)
                for ii, cc in enumerate(est_classes):
                    p[:, ii] = (preds == cc).astype(float)

            # Map estimator class columns to global class ordering
            # If estimator has attribute classes_, use it; otherwise infer from columns (rare)
            if hasattr(est_j, "classes_"):
                est_classes = np.asarray(est_j.classes_)
            else:
                # assume columns correspond to sorted unique labels seen in p
                est_classes = np.unique(y)

            # Build prob matrix aligned to self.classes_
            P_aligned = np.zeros((p.shape[0], C), dtype=float)
            for k, cls in enumerate(self.classes_):
                # find if this cls exists in est_classes
                mask = np.where(est_classes == cls)[0]
                if mask.size == 1:
                    P_aligned[:, k] = p[:, mask[0]]
                else:
                    # estimator never predicted class cls — leave zeros (effectively low confidence)
                    P_aligned[:, k] = 0.0

            # compute hard predictions for F1 calculation
            preds_aligned = self.classes_[np.argmax(P_aligned, axis=1)]

            # per-class F1: use binary labels vs predicted equality
            for i, cls in enumerate(self.classes_):
                f1 = f1_score(y == cls, preds_aligned == cls)
                smoothed = (f1 + self.epsilon) / (1.0 + self.epsilon)
                weights[j, i] = smoothed

            # global accuracy multiplier (optional)
            accs[j] = accuracy_score(y, preds_aligned)

        # optional multiplication by accuracy
        if self.weight_by_accuracy:
            # normalize accs to [0,1], add tiny epsilon
            accs = (accs - accs.min()) / (accs.max() - accs.min() + 1e-12)
            weights = weights * accs[:, None]

        # Normalize weights: for numerical stability, scale each estimator's vector to sum=1
        # so each estimator contributes a normalized class-weight distribution
        row_sums = weights.sum(axis=1, keepdims=True) + 1e-12
        weights = weights / row_sums

        self.class_weights_ = weights
        self.global_acc_ = accs

        if self.verbose:
            print(f"[SmoothedVoter] fitted {M} estimators; classes={self.classes_.shape[0]}")
            print(f"[SmoothedVoter] weights mean {weights.mean():.4f}, std {weights.std():.4f}")

        return self

    # -------------------------
    def predict_proba(self, X):
        """
        Weighted-probability voting (class-conditional).
        Returns probabilities aligned to self.classes_.
        """
        if self.class_weights_ is None:
            raise RuntimeError("Call fit(DSEL_X, DSEL_y) before predict_proba.")

        # Prepare per-estimator probabilities aligned to global classes
        X = np.array(X)
        M = len(self.estimators_)
        C = len(self.classes_)
        N = X.shape[0]
        proba_sum = np.zeros((N, C), dtype=float)

        for j, est in enumerate(self.estimators_):
            # get proba aligned
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(X)
            elif hasattr(est, "decision_function"):
                df = est.decision_function(X)
                if df.ndim == 1:
                    df = np.vstack([-df, df]).T
                p = _softmax(df, axis=1)
            else:
                # fall back to hard predictions
                preds = est.predict(X)
                p = np.zeros((N, len(np.unique(preds))), dtype=float)
                est_classes = np.unique(preds)
                for ii, cc in enumerate(est_classes):
                    p[:, ii] = (preds == cc).astype(float)

            # align columns
            if hasattr(est, "classes_"):
                est_classes = np.asarray(est.classes_)
            else:
                est_classes = self.classes_

            P_aligned = np.zeros((N, C), dtype=float)
            for k, cls in enumerate(self.classes_):
                mask = np.where(est_classes == cls)[0]
                if mask.size == 1:
                    P_aligned[:, k] = p[:, mask[0]]
                else:
                    P_aligned[:, k] = 0.0

            # multiply columns by estimator's per-class weights and accumulate
            proba_sum += P_aligned * self.class_weights_[j][None, :]

        # Normalize rows to be probabilities
        proba_sum = proba_sum / (proba_sum.sum(axis=1, keepdims=True) + 1e-12)
        return proba_sum

    # -------------------------
    def predict(self, X):
        if self.voting == "soft":
            proba = self.predict_proba(X)
            idx = np.argmax(proba, axis=1)
            return self.classes_[idx]
        else:
            # weighted hard voting: for each estimator, add its per-class weight to the predicted class column
            M, C = self.class_weights_.shape
            N = X.shape[0]
            vote_scores = np.zeros((N, C), dtype=float)
            for j, est in enumerate(self.estimators_):
                preds = est.predict(X)
                # map predicted class to index
                for i, cls in enumerate(self.classes_):
                    vote_scores[:, i] += (preds == cls).astype(float) * self.class_weights_[j, i]
            idx = np.argmax(vote_scores, axis=1)
            return self.classes_[idx]

    # -------------------------
    def diagnostics(self, top_k=10):
        """Print weight diagnostics and top-k per-estimator best/worst classes."""
        if self.class_weights_ is None:
            print("Model not fitted yet.")
            return

        weights = self.class_weights_
        M, C = weights.shape

        print("\n" + "="*70)
        print("               CLASS-CONDITIONAL WEIGHT DIAGNOSTICS (IMPROVED)")
        print("="*70)
        print(f"Estimators: {M}, Classes: {C}")
        print(f"Global accuracy per estimator (if computed):")
        if self.global_acc_ is not None:
            for j, (name, _) in enumerate(self.estimators):
                print(f"  {j:02d} {name:20s} acc={self.global_acc_[j]:.3f}, mean weight={weights[j].mean():.4f}")

        print("\nPer-estimator top/bottom classes by weight:")
        for j, (name, _) in enumerate(self.estimators):
            sorted_idx = np.argsort(-weights[j])  # descending
            top = sorted_idx[:top_k]
            bottom = sorted_idx[-top_k:]
            print(f"\nEstimator {j:02d} {name}:")
            print("  Top classes:", list(self.classes_[top]))
            print("  Bottom classes:", list(self.classes_[bottom]))

        print("\nOverall weight stats:")
        print(f"  Mean weight:  {weights.mean():.4f}")
        print(f"  Std:          {weights.std():.4f}")
        print(f"  Min weight:   {weights.min():.4f}")
        print(f"  Max weight:   {weights.max():.4f}")
        print("="*70 + "\n")
