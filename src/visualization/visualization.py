def draw_ellipse(position, covariance, ax=None, **kwargs):
  
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)


def plt_graph(model, X, label=True, ax=None):
    if model == GaussianMixture():
        ax = ax or plt.gca()
        labels = model.fit(X).predict(X)
        if label:
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='jet', zorder=2)
        else:
            ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
        ax.axis('equal')

        w_factor = 0.2 / model.weights_.max()
        for pos, covar, w in zip(model.means_, model.covariances_, model.weights_):
            draw_ellipse(pos, covar, alpha=w * w_factor)
    else:
        ax = ax or plt.gca()
        labels = model.fit(X).predict(X)
        if label:
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='jet', zorder=2)
        else:
            ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
        ax.axis('equal')

def plt_3d_tsne(mod,X_split):
    tsne = TSNE(n_components=3, init = "random", learning_rate = "auto", random_state=99)
    X_reduceds = tsne.fit_transform(X_split)
    tsne_pred = mod.fit(X_reduceds).predict(X_reduceds)

    # plotting

    X_reduceds = pd.DataFrame(X_reduceds, columns=(['TSNE 1', 'TSNE 2', 'TSNE 3']))
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_reduceds['TSNE 1'],X_reduceds['TSNE 2'],X_reduceds['TSNE 3'], c=tsne_pred)
    ax.set_title("3D projection of the clusters")

    
def plt_3d_pca(mod,X_split):
    pca = PCA(n_components=3, random_state=99)
    X_reduceds = pca.fit_transform(X_split)
    pca_pred = mod.fit(X_reduceds).predict(X_reduceds)

    # plotting

    X_reduceds = pd.DataFrame(X_reduceds, columns=(['PCA 1', 'PCA 2', 'PCA 3']))
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_reduceds['PCA 1'],X_reduceds['PCA 2'],X_reduceds['PCA 3'], c=pca_pred)
    ax.set_title("3D projection of the clusters")
    
    
def plt_jet(model, X, figsizes=(10,8)):
    labels = model.fit(X).predict(X)
    plt.figure(figsize=figsizes)
    plt.scatter(X[:, 0], X[:, 1], c = labels.astype(np.int8), cmap="jet", alpha=0.5)
    plt.colorbar()
    plt.show()

    ds_columns = df.columns
plt.figure(figsize=(18,15))

for i in range(len(ds_columns)):
    plt.subplot(6 , 3, i+1)
    sns.histplot(x=df[ds_columns[i]], kde=True)
    plt.title("Displot is {}".format(ds_columns[i]))
    plt.tight_layout()

plt.figure(figsize=(13,13))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap = plt.cm.Reds)
plt.show()

inertia = []
list_num_clusters = list(range(1,16))
for num_clusters in list_num_clusters:
    km = KMeans(n_clusters=num_clusters,init = "k-means++",algorithm="auto", random_state=99)
    km.fit(X_train)
    inertia.append(km.inertia_)
    
plt.plot(list_num_clusters,inertia)
plt.scatter(list_num_clusters,inertia, c = "orange", marker = "o")
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')


inertia = []
list_num_clusters = list(range(1,16))
for num_clusters in list_num_clusters:
    km = KMeans(n_clusters=num_clusters,init = "k-means++",algorithm="auto", random_state=99)
    km.fit(X_reduced)
    inertia.append(km.inertia_)
    
plt.plot(list_num_clusters,inertia)
plt.scatter(list_num_clusters,inertia, c="green", marker="o")
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title("Elbow Methoht for KMeans")
plt.legend()
plt.show()

plt_jet(kmeans_elb,X_reduced)
plt_graph(kmeans_elb,X_reduced)
plt_3d_tsne(kmeans_elb, X_train)

plt_jet(kmeans_pca, X_pca_reduced)
plt_3d_pca(kmeans_pca, X_train)

###MİNİBATCH Clustering
inertia_batch = []
n_cluster_batch = list(range(1, 16))

plt.figure(figsize=(10, 6)) 

for cluster_num_batch in n_cluster_batch:
    minibatch = MiniBatchKMeans(n_clusters=cluster_num_batch, batch_size=100, init="k-means++")
    minibatch.fit(X_train)
    inertia_batch.append(minibatch.inertia_)

plt.plot(n_cluster_batch, inertia_batch, label='Inertia')
plt.scatter(n_cluster_batch, inertia_batch, c='red', marker='o')
plt.xlabel("Number of Cluster")
plt.ylabel("Inertia")
plt.title("Elbow Method for MiniBatchKMeans")
plt.legend()
plt.show()

plt_jet(minibatch, X_reduced)
plt_3d_tsne(minibatch, X_train)
plt_jet(kmeans_mini_pca, X_pca_reduced)
plt_3d_pca(kmeans_mini_pca, X_train)

### DBSCAN Clustering

plt.rcParams['figure.figsize'] = (20,15)
unique_labels = set(dbscn.labels_)
n_labels = len(unique_labels)
cmap = plt.cm.get_cmap('brg', n_labels)
for l in unique_labels:
    plt.scatter(
        X_reduced[dbscn.labels_ == l, 0],
        X_reduced[dbscn.labels_ == l, 1],
        c=[cmap(l) if l >= 0 else 'Black'],
        marker='ov'[l%2],
        alpha=0.75,
        s=100,
        label=f'Cluster {l}' if l >= 0 else 'Noise')
plt.legend(bbox_to_anchor=[1, 1])
plt.show()
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

### GAUSSİAN MİXTURE Clustering 

plt_jet(gmm, X_reduced)
plt_3d_tsne(gmm,X_train)
plt_graph(gmm, X_reduced)
plt_jet(gmm_pca, X_pca_reduced)
plt_graph(gmm_pca, X_pca_reduced)
plt_3d_pca(gmm, X_train)

### MEAN-SHIFT Clustering

plt_jet(ms, X_reduced)
plt_graph(ms, X_reduced)

###  Agglomerative Clustering (Hierarchical)

plt.figure(figsize=(30,15))
dendrogram = sch.dendrogram(sch.linkage(X_train, method = 'ward'))
plt.title('Training Set')
plt.xlabel('X Values')
plt.ylabel('Distances')
plt.show()

plt.figure(figsize=(30,15))
dendrogram = sch.dendrogram(sch.linkage(X_test, method = 'ward'))
plt.title('Test Set')
plt.xlabel('X Value')
plt.ylabel('Distances')
plt.show()

