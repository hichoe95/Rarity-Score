import numpy as np
from sklearn.mixture import GaussianMixture
import gc

def load_gmm(path):
	means = np.load(gmm_name + '_means.npy')
	covar = np.load(gmm_name + '_covariances.npy')
	real_gm = GaussianMixture(n_components = len(means), covariance_type='full')
	real_gm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
	real_gm.weights_ = np.load(gmm_name + '_weights.npy')
	real_gm.means_ = means
	real_gm.covariances_ = covar

	return real_gm