import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as ss
from sklearn.utils import shuffle
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from treeple import ObliqueRandomForestClassifier, PatchObliqueRandomForestClassifier

class PvalueExplainer:
    """
    A class for assessing feature importance in datasets using permutation testing
    with oblique random forest classifiers (MORF or SPORF).        
    """
    
    @staticmethod
    def train_model(ii, X, y, clf):
        """
        Train a classifier and return the feature importance and OOB decisions.

        Parameters:
        -----------
        ii: int
            The random seed.
        X: array-like of shape (n_samples, n_features)
            The training input samples.
        y: array-like of shape (n_samples,)
            The target values.
        clf: str
            The classifier type, either "MORF" or "SPORF".

        Returns:
        --------
        tuple
            (feature_importance, padded_oob_decisions)
            feature_importance: array-like of shape (n_features,)
                The feature importance.
            padded_oob_decisions: array-like of shape (n_samples, n_classes)
                The OOB decisions.
        """
        rng = np.random.default_rng(ii)
        bootstrap_idx = rng.choice(len(X), size=len(X), replace=True)
        
        if clf == "MORF":
            orf = PatchObliqueRandomForestClassifier(
                n_estimators=1,
                max_patch_dims=np.array((5, 2)),
                data_dims=np.array((28, 28)),
                n_jobs=1,
                max_features='sqrt',
                bootstrap=False,
                verbose=1,
                oob_score=False,
                random_state=ii,
            )
        elif clf == "SPORF":
            orf = ObliqueRandomForestClassifier(
                n_estimators=1,
                n_jobs=1,
                max_features='sqrt',
                bootstrap=False,
                verbose=1,
                oob_score=False,
                random_state=ii,
            )
        else:
            raise ValueError(f"Classifier type '{clf}' not supported yet.")

        orf.fit(X[bootstrap_idx, :], y[bootstrap_idx])
        oob_idx = np.setdiff1d(np.arange(len(y)), bootstrap_idx)
        oob_decisions = orf.predict_proba(X[oob_idx, :])
        padded_oob_decisions = np.zeros((len(y), oob_decisions.shape[1]))
        padded_oob_decisions[oob_idx, :] = oob_decisions
        
        return orf.feature_importances_, padded_oob_decisions
    
    @staticmethod
    def compute_ranks(feature_importance):
        """Precompute ranks using 'max' method to match original behavior."""
        return np.apply_along_axis(lambda x: ss.rankdata(1 - x, method='max'), axis=1, arr=feature_importance)

    @staticmethod
    def statistics(ranks, idx, n_estimators=100):
        """
        Calculate the feature importance test statistic.
        Ensures correct rank ordering and functionality preservation.
        """
        stat = np.zeros(ranks.shape[1])
        
        for ii in range(n_estimators):
            r = ranks[idx[ii]]
            r_0 = ranks[idx[n_estimators + ii]]
            stat += (r_0 > r) * 1  # Boolean comparison
            
        stat /= n_estimators
        return stat

    @staticmethod
    def perm_stat(ranks, n_estimators=100):
        """
        Helper function that calculates the null distribution.
        """
        rng = np.random.default_rng()
        idx = rng.permutation(2 * n_estimators)
        return PvalueExplainer.statistics(ranks, idx, n_estimators)

    @classmethod
    def test(cls, feature_importance, n_repeats=1000, n_jobs=-1, n_est=100):
        """
        Permutation test to compare real vs shuffled feature importance.
        """
        # Precompute ranks once using the correct method
        ranks = cls.compute_ranks(feature_importance)

        # Compute actual statistic
        stat = cls.statistics(ranks, np.arange(2 * n_est), n_est)

        # Parallel computation of null distribution
        null_stat = np.array(Parallel(n_jobs=n_jobs)(
            delayed(cls.perm_stat)(ranks, n_estimators=n_est)
            for _ in tqdm(range(n_repeats), desc="Calculating null distribution")
        ))

        # Compute p-values
        count = np.sum(null_stat >= stat, axis=0)
        p_val = (1 + count) / (1 + n_repeats)

        return stat, p_val
    
    @classmethod
    def get_p(cls, feat_imp_all, feat_imp_all_rand, n_repeats=5000):
        """Calculate p-values with multiple testing correction."""
        _, p = cls.test(
            np.concatenate((feat_imp_all, feat_imp_all_rand)),
            n_repeats=n_repeats,
            n_jobs=-1,
            n_est=len(feat_imp_all)
        )

        # Apply Bonferroni-Holm correction
        p_corrected = multipletests(p, method="holm")[1]
        return p_corrected
    
    @classmethod
    def feat_imp_test(cls, X, y, n_est, n_rep, clf):
        """
        Main method to test for significant features.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        n_est : int
            Number of estimators to use.
        n_rep : int
            Number of repetitions for the permutation test.
        clf : str
            Classifier type, either "MORF" or "SPORF".
            
        Returns:
        --------
        corrected_p : array-like of shape (n_features,)
            Corrected p-values for each feature.
        """
        print(f"Training {n_est} models on original data...")
        results = Parallel(n_jobs=-1)(
            delayed(cls.train_model)(ii, X, y, clf) for ii in tqdm(range(n_est))
        )
        feat_imp_all, _ = zip(*results)
        
        print(f"Training {n_est} models on shuffled data...")
        y_shuffled = shuffle(y, random_state=0)
        results = Parallel(n_jobs=-1)(
            delayed(cls.train_model)(ii, X, y_shuffled, clf) for ii in tqdm(range(n_est))
        )
        feat_imp_all_rand, _ = zip(*results)
        
        print(f"Computing p-values with {n_rep} repetitions...")
        corrected_p = cls.get_p(np.array(feat_imp_all), np.array(feat_imp_all_rand), n_rep)
        return corrected_p
    
    @classmethod
    def get_significant_features(cls, X, y, n_est=100, n_rep=5000, clf="SPORF", alpha=0.05):
        """
        Find significant features (p < alpha) and return their indices.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        n_est : int, default=100
            Number of estimators to use.
        n_rep : int, default=5000
            Number of repetitions for the permutation test.
        clf : str, default="SPORF"
            Classifier type, either "MORF" or "SPORF".
        alpha : float, default=0.05
            Significance level threshold.
            
        Returns:
        --------
        significant_features : array-like
            Boolean array indicating significant features.
        p_values : array-like
            Corrected p-values for each feature.
        """
        p_values = cls.feat_imp_test(X, y, n_est, n_rep, clf)
        significant_features = p_values < alpha
        
        print(f"Found {np.sum(significant_features)} significant features out of {len(significant_features)}")
        return significant_features, p_values
    
    @classmethod
    def filter_features(cls, X, significant_features):
        """
        Filter X to only include significant features.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        significant_features : array-like of shape (n_features,)
            Boolean array indicating significant features.
            
        Returns:
        --------
        X_filtered : array-like
            X with only significant features.
        """
        return X[:, significant_features]