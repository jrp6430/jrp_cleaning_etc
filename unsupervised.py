import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import kneed
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# *** A NOTE ON K MEANS CLUSTERING *** --------------------------------------------------------------------------------

# 1. Establish the number of clusters (k)
# 2. K points are randomly selected to be the initial centroids for the algorithm
# 3. The Euclidean distance from all points to each centroid is calculated
# 4. Points are assigned to each centroid based on which one they are closest to
# 5. For each centroid, the coordinates belonging to them are averaged, and this average becomes the new centroid
# 6. Repeat steps 3-5 until a point of convergence is reached.

# A caveat of this method is that it has the potential to convergence to a local minimum. To avoid this, choosing an
# appropriate k value is essential.

# A method of evaluating clustering is through inertia. In this, the distance between each point and its centroid is
# calculated, squared, and then summed, representing the spread of clustering. To find the optimal k for clustering,
# calculate the inertia for multiple values of k and find the point where an increase in clusters has a diminishing
# impact on decreasing spread.

# K means plus is a modification to the original algorithm in terms of initial centroid seeding. Instead of choosing all
# k randomly, one is picked randomly at first. Then D(x) is calculated for each point, which is the distance for it to
# the nearest centroid. We then create a probability distribution proportional to D(x)^2, such that points farther away
# from centers are more likely to be chosen as the next centroid. Repeat this until all k have been chosen. This makes
# it less likely to encounter a local minimum, as  opposed randomly chosen centroids all in the same proximity resulting
# in weaker clustering.

def optimal_kmeans_plus(data, plot_dim):
    n = len(data)
    max_k = int(np.sqrt(n))
    inertia = np.zeros(max_k)
    cluster_iterator = range(1, max_k + 1)
    # indexing through k=1 to k=kmax, extracting inertia for each k
    for i in cluster_iterator:
        result = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=17).fit(data)
        inertia[i - 1] = result.inertia_
    # finding elbow, then performing kmeans at the elbow value
    knee = kneed.KneeLocator(cluster_iterator, inertia, curve='convex', direction='decreasing')
    elbow = knee.knee
    print('The chosen number of clusters based off of the elbow method is %.1f' % elbow)
    best = KMeans(n_clusters=elbow, init='k-means++', n_init=15, random_state=17).fit(data)
    labels = best.labels_

    # plotting if plot_dim is 2 or 3, ending if not

    if isinstance(plot_dim, int) and not isinstance(plot_dim, bool):
        if plot_dim > 2:
            fig1 = plt.figure(1)
            ax = fig1.add_subplot(projection='3d')
        else:
            fig1 = plt.figure(1)
            ax = fig1.add_subplot()
        k_indices = [[]]
        for i in range(elbow - 1):
            k_indices.append([])
        for i in range(elbow):
            plotting_group = [[]]
            for go in range(plot_dim-1):
                plotting_group.append([])
            for j in range(len(labels)):
                if i == labels[j]:
                    point = data.iloc[j]
                    for q in range(plot_dim):
                        plotting_group[q].append(list(point)[q])
                        k_indices[i].append(j)
            if plot_dim == 3:
                ax.scatter(plotting_group[0], plotting_group[1], plotting_group[2])
            else:
                ax.scatter(plotting_group[0], plotting_group[1])
        ax.set_xlabel('Axis 1')
        ax.set_ylabel('Axis 2')
        if plot_dim > 2:
            ax.set_zlabel('Axis 3')
        ax.set_title('K-Means++ Clustering of Dataset, k=%.1f' % elbow)
        plt.show()
    return


# Like for tSNE, the usefulness of PCA (to me) is as a preprocessing technique, where high-parameter datasets are
# simplified to principal components which (in ascending) explain the most variance in the set
# Mathematically, principal component axes are the eigenvectors of the eigenvalues of the covariance matrix
# The larger the eigenvalue, the more variation the eigenvector, or PC captures.

# *** A NOTE ON PCA *** ----------------------------------------------------------------------------------------------
#
# This feature extraction method relies on the principle of eigenvalues and eigenvectors. For matrix A:
#
# Av = ev, where v is the eigenvector and e is an eigenvalue. For this to work, A must be a square matrix.
#
# When A is the covariance matrix of a dataset, the eigenvectors represent a vector direction that captures the largest
# spread of the data. This is proportional to its eigenvalue, where the largest eigenvalued eigenvector captures the
# most variation in the input dataset.
#
# So, each principal component is an eigenvector capturing variation in the data, and we want to project our points onto
# this new plane. This is done by using the unit vector of each eigenvector: u^T*x for each point x. For a dataset of
# D features, there will be D eigenvalues. We will select how many principal components used based on the amount
# of variance they explain, typically greater than 80%.
#
# A math consideration: |A - eI| = 0. This means that the determinant of the matrix difference between the covariance
# matrix and identity matrix * e must be zero. This is needed for when floating point errors arise in calculations.
#
# Due to the nature of eigenvectors being mutually perpendicular, PCA removes multi-collinearity from the set, so it can
# be useful as a preprocessor for machine learning algorithms. A pitfall however is that it is a linear method.

def pick_pca_transform(data):

    # defining variables, n of observations, max k value as sqrt(n)
    features = []
    n = len(data)
    for col in data.columns:
        features.append(col)
    m = len(features)

    # standard scaling (z score) prior to pca and clustering
    ss = StandardScaler()
    ss.fit(data)
    normalized = ss.transform(data)

    # determining PCA suitability

    pass

    # setting maximum number of pcs as the minimum choice between n and # of parameters
    if m > n:
        start_comp = n
    else:
        start_comp = m

    # run PCA with maximum number of pcs and fit it to normalized data
    pca = PCA(n_components=start_comp)
    pca.fit(normalized)
    pc_choice = 0
    # from 0 to the max PC, cumulatively sum the amount of variance (starting with PC1, then PC1+PC2, etc.)
    for i in range(0, start_comp):
        var = sum(pca.explained_variance_ratio_[0:i])
        # when the cumulative variance achieves 80% explained, perform PCA again with that number of PCs
        if var >= 0.8:
            pc_choice = i+1
            pca = PCA(n_components=pc_choice)
            pca.fit(normalized)
            # end the loop to prevent it from adding more PCs
            break
    print('The chosen number of PCs is %.1f' % pc_choice)
    pca_to_pipe = PCA(n_components=pc_choice)
    # transform the normalized data on the final PC selection, store variance explained and loadings
    pc_variance = sum(pca.explained_variance_ratio_ * 100)
    print('The variance explained by this choice is %.2f percent' % pc_variance)

    return pca_to_pipe

# moving on to another unsupervised dimensionality reduction tool - tSNE (T-Distributed Stochastic Neighbor Embedding)


# tsne (to me) seems like a pre-processing tool for other algorithms like random forest, where the reduction in d
# speeds up computation/ processing time.

# *** A NOTE ON tSNE *** ---------------------------------------------------------------------------------------------
#
# This nonlinear  feature extraction method creates a Gaussian distribution in higher dimensions on pairs of points,
# such that similar points have a higher chance of being chosen. Then, the algorithm replicates this distribution on
# a lower dimensional space until KL divergence is minimzed. Here are the steps in greater detail:
#
# Step 1: Calculate pairwise similarities between point "xi" and all other points. Consider one point xj:
#
# p(i,j) = (exp(-||xi - xj||^2 / 2σi^2) / sigma[i != k]( exp(-||xi - xk||^2 / 2σi^2))
#
# Where || xi - point || is the Euclidean distance, σ is a parameter defined by the perplexity hyperparameter. As
# perplexity increases, the σ term does too, causing more points to be deemed similar to xi. The above equation can be
# interpreted the ratio of the similarity of xi to xj with respect to the cumulative similarity of xi with all other
# points. This is our high dimensional probability distribution, and is computed for all xj for given point i over all
# i centers.
#
# Step 2: Find the pairwise similarities in the lower dimensional student t distribution representation:
#
# q(i, j) = (1 + ||yi - yj||^2)^-1 / sigma[i != l]( (1 + ||yi - yl||^2)^-1) - for all j other points across all centers
#
# After this step, we will have two nxn similarity matrices, one for q and one for p.
#
# Step 3: With both high and low dimensional probability distributions for pairwise similarity, start iterating with
# gradient descent as determined by the KL-divergence optimization function. We do this with a cost function, the
# gradient of KL divergence between the two distributions.
#
# D(q, p) = sigma(p(x)*log(p(x)/q(x))) - the smaller this is the closer p(x) is to q(x)
#
# We want to adapt the lower dimensional dataset such that q(i, j) becomes as close as possible to p(i, j)
#
# C = sigma sigma [ all i and j]( p(i, j)*log(p(i, j)/q(i, j)) )
#
# ΔC(yi) = 4*sigma((p(i, j) - q(i, j))*(yi - yj)*(1 + ||yi - yj||^2)) - for all j other points across all centers i
#
# Remember, the gradient is the direction of steepest change in the base function, which is C for our purposes. SO,
# C will decrease fastest when point i is moved in the direction of the negative gradient ΔC. So for one iteration
# of gradient descent for point i:
#
# i+1 = i - αΔC(i), where α is the learning rate parameter that can be tuned. We repeat this until i converges!
# This is done for all points in the dataset to transform to the lower dimensional space.


def tsne_bot(data_df, normalize, target):
    # determine sample size
    n = len(data_df)

    # first, normalize with z score. Turn this off if input is already normalized, or doesn't need it (PCA)
    if normalize:
        ss = StandardScaler()
        ss.fit(data_df)
        normalized = ss.transform(data_df)
    else:
        normalized = data_df

    # optimize learning rate based on sample size: Kobak & Berens 2019. Corresponds with the early exaggeration def (12)
    possible_learn = [200, n/12]
    learning_rate = max(possible_learn)

    # first optimize perplexity: reflects the number of nearest neighbors that is used in other manifold algorithms
    # Use S parameter detailed by Cao and Wang: Automatic Selection of t-SNE perplexity
    # This accounts for the fact that KL divergence will naturally decrease with increasing perplexity, always favoring
    # a higher value. The offset is the perplexity's relationship to the sample size, as the max perplexity value is n

    # n/3 taken from OpenTSNE documentation
    perplexity = np.arange(5, n/3, n/30)
    s_param = []

    # offset part of S will not change regardless of perplexity
    offset = np.log10(n)*(1/n)

    # calculate S using KL divergence for each perplexity
    for i in perplexity:
        model = TSNE(n_components=2, perplexity=i, learning_rate=learning_rate, random_state=17)
        model.fit(normalized)
        kl1 = model.kl_divergence_
        s_param.append((2*kl1) + (offset*i))

    # choosing the perplexity value associated with minimal S
    best_perplexity = perplexity[np.argmin(s_param)]
    print('The chosen perplexity value is %.1f' % int(best_perplexity))

    # now that perplexity is fixed, range through n_iter and minimize KL divergence
    n_iter = [500, 1000, 2000, 3000, 5000, 7500, 10000]
    working_kl = []
    for j in n_iter:
        the_model = TSNE(n_components=2, perplexity=best_perplexity, learning_rate=learning_rate, n_iter=j,
                         random_state=17)
        the_model.fit(normalized)
        kl2 = the_model.kl_divergence_
        working_kl.append(kl2)

    # best n is the value that minimizes KL divergence
    best_n = n_iter[np.argmin(working_kl)]
    print('The number of iterations that minimizes KL divergence is %.1f' % best_n)

    # fit with tuned parameters and transform data to tSNE space
    best_model = TSNE(n_components=2, perplexity=best_perplexity, learning_rate=learning_rate, n_iter=best_n,
                      random_state=17)
    tsne_transformed = best_model.fit_transform(normalized)

    # dealing with non-numerical targets in plotting tsne 1+2, mapping unique classifiers to numbers
    if type(target[0]) is str:
        classes = np.unique(target)
        for i in range(len(classes)):
            for j in range(len(target)):
                if target[j] == classes[i]:
                    target[j] = i

    # visualizing the results of the operation
    plt.scatter(tsne_transformed[:, 0], tsne_transformed[:, 1], c=target)
    plt.xlabel('First tSNE')
    plt.ylabel('Second tSNE')
    plt.title('Dimensionality Reduction with t-SNE, perplexity = %.1f' % int(best_perplexity))
    plt.show()
    return
