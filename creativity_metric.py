import numpy as np
import torch

from sklearn.cluster import KMeans as kmeans
from improved_precision_recall import compute_pairwise_distances



class CREATIVITY(object):
	def __init__(self, real_features, fake_features):

		self.real_features = real_features 
		self.fake_features = fake_features

		self.num_reals = real_features.shape[0]
		self.num_fakes = fake_features.shape[0]

		print('Preprocessing pairwise diatances ...')
		self.real2fake_distances = compute_pairwise_distances(real_features, fake_features)
		self.real2real_distances = compute_pairwise_distances(real_features)

		self.real2real_sorted = np.sort(self.real2real_distances, axis = 1)
		self.real2real_sorted_ids = self.real2real_distances.argsort(axis = 1)

		# for clustering
		self.num_cluster = 0
		self.modes = None
		self.sample_mode_ids = None

		self.mode2mode_distances = None
		self.mode2fake_distances = None
		self.mode2mode_sorted = None
		self.mode2mode_sorted_ids = None


	def is_in_ball(k = 3, samples = None, cluster = False):
		""" Compute the differences between radii of kNN real balls and distances
			for judging whether they are in each kNN real ball or not.

			args:
				k (int): real ball's size is distance between reference real sample 
						 and k th nearest real sample.
				samples (np.array, num_samples * embed_dim): embedded generation samples
			return:
				dist_radi (np.array, num_reals * num_samples): each element means 
						 (radii of real - (distance between real and sample)).
						 if it is larger than 0, it is out of the real ball.
				r (np.array, num_reals * 1): radii of each kNN real balls
				out_ball_ids (np.array, num_out_ball_samples): indices of samples outside of balls.
		"""

		if samples is None:
			samples = self.fake_features
			real2samples_distances = self.real2fake_distances if !cluster else self.mode2fake_distances
		else:
			real2samples_distances = compute_pairwise_distances(self.real_features, samples) if !cluster else \
			compute_pairwise_distances(self.modes, samples)

		r = self.real2real_sorted[:,k:k+1] if !cluster else self.mode2mode_sorted[:, k:k+1]

		dist_radi = (r.repeat(samples.shape[0], axis = 1) - real2samples_distances)
		out_ball_ids = np.where((dist_radi > 0).any(axis = 0) == False)[0]

		return dist_radi, r, out_ball_ids

	def metric1(self, k = 3, samples = None):
		""" The larger the real ball's size, the rare the real sample would be.

			args:
				k (int): real ball's size is distance between reference real sample 
						 and k th nearest real sample.
				samples (np.array, N * embed_dim): embedded generation samples
			return:
				scores (np.array, num_samples): scores of each samples that are not sorted.
				scores_ids (np.array, num_in_ball_samples): for samples in valid reall balls,
					  sorted indices in decreasing order.
		"""

		samples = self.fake_features if samples is None else samples
		in_ball_dist, r, out_ball_ids = self.is_in_ball(samples, k = k)

		num_out_ball = len(out_ball_ids)
		valid_real_balls = (in_ball_dist>0)

		scores = np.zeros(samples.shape[0])

		for i in range(samples.shape[0]):
			if i not in out_ball_ids:
				scores[i] = r[valid_real_balls[:,i]].min()

		scores_ids = (-scores).argsort()[:samples.shape[0] - num_out_ball]

		return scores, scores_ids

	def metric2(self, k = 3, samples = None):
		""" When there are multiple real balls that contain a fake sample,
			the more apart the real balls are, the more creative the fake sample would be.

			args:
				k (int): real ball's size is distance between reference real sample 
						 and k th nearest real sample.
				samples (np.array, N * embed_dim): embedded generation samples
			return:
				scores (np.array, num_samples): scores of each samples that are not sorted.
				scores_ids (np.array, num_in_ball_samples): for samples in valid reall balls,
					  sorted indices in decreasing order.
		"""

		samples = self.fake_features if samples is None else samples

		in_ball_dist, r, out_ball_ids = self.is_in_ball(samples, k = k)
		num_out_ball = len(out_ball_ids)

		valid_real_balls = (in_ball_dist > 0)
		reals_in_real_ball = self.real2real_sorted_ids[:, :k+1] # k * num_real_samples

		scores = np.zeros(samples.shape[0])

		for i in range(samples.shape[0]):
			if i not in out_ball_ids:
				real_samples = np.unique(reals_in_real_ball[valid_real_balls[:, i], :].flatten())
				scores[i] = self.real2real_distances[real_samples, :][:, real_samples].max()

		scores_ids = (-scores).argsort()[:samples.shape[0] - num_out_ball]

		return scores, scores_ids


	def clustering_kmeans(self, k = 3, num_cluster = 300, samples = None):
		"""
		"""
		samples = self.fake_features if samples is None else samples

		if self.num_cluster != num_cluster:
			cluster = kmeans(n_clusters = num_cluster).fit(real_manifold.features)
			self.modes = cluster.cluster_centers_
			self.sample_mode_ids = cluster.labels_

			self.mode2mode_distances = compute_pairwise_distances(self.modes)
			self.mode2fake_distances = compute_pairwise_distances(self.modes, samples)
			self.mode2mode_sorted = np.sort(self.mode2mode_distances, axis = 1)
			self.mode2mode_sorted_ids = self.mode2mode_distances.argsort(axis = 1)


	def metric3(self, k = 15, num_cluster = 300, samples = None):

		samples = self.fake_features if samples is None else samples

		self.clustering_kmeans(k = k, num_cluster = num_cluster, samples = samples)

		in_ball_dist, r, out_ball_ids = self.is_in_ball(samples, k = k, cluster = True)
		num_out_ball = len(out_ball_ids)

		valid_mode_balls = (in_ball_dist>0)
		modes_in_mode_ball = self.mode2mode_sorted_ids[:, :k+1]

		scores = np.zeros(samples.shape[0])

		for i in range(samples.shape[0]):
			if i not in out_ball_ids:
				modes = np.unique(modes_in_mode_ball[valid_mode_balls[:, i], :].flatten())
				scores[i] = self.mode2mode_distances[modes, :][:, modes].max()

		scores_ids = (-scores).argsort()[:samples.shape[0] - num_out_ball]

		return scores, scores_ids

	def metric4(self, k = 15, num_cluster = 300, samples = None):

		samples = self.fake_features if samples is None else samples

		self.clustering_kmeans(k = k, num_cluster = num_cluster, samples = samples)

		in_ball_dist, _, __ = self.is_in_ball(k = k, samples = self.real_features, cluster = True)
		num_reals_in_mode_ball = (in_ball_dist > 0).sum(axis = 1)

		in_ball_dist_sample, _, out_ball_ids = self.is_in_ball(k = k, cluster = True)
		num_out_ball = len(out_ball_ids)

		modes_ids, samples_ids = np.where(in_ball_dist_sample > 0)
		_, cnt = np.unique(samples_ids, return_counts = True)

		scores = np.zeros(samples.shape[0])
		for i in range(samples.shape[0]):
			mode_ids = modes_ids[cnt[i] : cnt[i+1]]
			scores[i] = (1. / num_reals_in_mode_ball[mode_ids]).sum()

		scores_ids = (-scores).argsort()[:samples.shape[0] - num_out_ball]

		return scores, scores_ids

























