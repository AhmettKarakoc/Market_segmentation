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