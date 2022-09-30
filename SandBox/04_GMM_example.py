from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
# Plot the data with K Means Labels
from GMM_utils import *


X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
fig, axs = plt.subplots(1,2)
axs[0].scatter(X[:, 0], X[:, 1], c=y_true, s=40, cmap='viridis')
axs[0].set_title('true')

gmm = GaussianMixture(n_components=4).fit(X)
plot_gmm(gmm, X)
# axs[1].scatter(X[:, 0], X[:, 1], c=gmm.predict(X), s=40, cmap='viridis')
# axs[1].set_title('estimated')