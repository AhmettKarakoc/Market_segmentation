## Modeling and Model Improvement
scaler = StandardScaler()
df = scaler.fit_transform(df)

X_train, X_test= train_test_split(df, test_size=0.8, shuffle = True ,random_state=99)

### KMEANS Clustering
silhouette_train_scores = []
silhouette_test_scores = []
max_clusters = 15

for i in range(2, max_clusters + 1):
    km = KMeans(n_clusters=i, init="k-means++",algorithm="auto", random_state=99)
    labels = km.fit_predict(X_train)
    score = silhouette_score(X_train, labels)
    silhouette_train_scores.append(score)
    
    labels_test = km.fit_predict(X_test)
    score_test= silhouette_score(X_test, labels_test)
    silhouette_test_scores.append(score_test)
    cluster_numbers = list(range(2, max_clusters + 1))

df_silhouette = pd.DataFrame({'Train_Silhouette': silhouette_train_scores, 'Test_Silhouette': silhouette_test_scores}, index=cluster_numbers)
df_silhouette.index.name = 'cluster_num'
df_silhouette

kmeans_elb = KMeans(n_clusters= 4 , init = "k-means++", algorithm = "auto", random_state=99)
kmeans_elb.fit(X_reduced)

silhouette_score_elb_train = silhouette_score(X_train, kmeans_elb.fit_predict(X_train))
silhouette_score_elb_test = silhouette_score(X_test , kmeans_elb.fit_predict(X_test))

print("Silhouette Score for Train: {}".format(silhouette_score_elb_train))
print("Silhouette Score for Test: {}".format(silhouette_score_elb_test))

kmeans_pca = KMeans(n_clusters= 4 , init = "k-means++", algorithm = "auto", random_state=99)

### MİNİBATCH Clustering
silhouette_score_batch_train = []
silhouette_score_batch_test = []
n_cluster_batch= range(2,16)
for num_cluster in n_cluster_batch:
    minibatch = MiniBatchKMeans(n_clusters = num_cluster, batch_size= 100, init="k-means++")
    labels = minibatch.fit_predict(X_train)
    silhouette_score_batch = silhouette_score(X_train,labels)
    silhouette_score_batch_train.append(silhouette_score_batch)
    
    labels_test = minibatch.fit_predict(X_test)
    silhouette_score_test = silhouette_score(X_test,labels_test)
    silhouette_score_batch_test.append(silhouette_score_test)
    
cluster_num = list(range(2,16))
df_silhouette_batch = pd.DataFrame({"Train_Silhouette":silhouette_score_batch_train, "Test_Silhouette":silhouette_score_batch_test},index=cluster_num)
df_silhouette_batch.index.name = "Cluster Num"
df_silhouette_batch
minibatch = MiniBatchKMeans(n_clusters=7, batch_size = 100, init="k-means++")
kmeans_mini_pca = MiniBatchKMeans(n_clusters = 7, batch_size=100, random_state=99)

### DBSCAN Clustering

dbscn = DBSCAN(eps =6 , min_samples = 38)
dbscn.fit(X_reduced)
print(f'DBCSANd found {len(set(dbscn.labels_)-set([-1]))} clusters and {(dbscn.labels_ ==-1).sum()} point of noise.')
#### The percentage of noise
print(f"{100*(dbscn.labels_ == -1).sum()/len(dbscn.labels_)}%")

### GAUSSİAN MİXTURE Clustering 

gmm = GaussianMixture(n_components = 10, covariance_type = "full", random_state=99)
gmm_pca = GaussianMixture(n_components = 10, covariance_type ="full", random_state=99)

### MEAN-SHIFT Clustering

#bandwidth = estimate_bandwidth(X_train)`estimate_bandwidth` function is a method for automatically estimating the bandwidth for the Mean Shift algorithm.
#I manually adjusted the bandwidth because it didn't perform well on my dataset.
ms = MeanShift(bandwidth=12 , bin_seeding=True)
ms.fit(X_train)

###  Agglomerative Clustering (Hierarchical)

train_pred = []
test_pred = []
n_cluster_agg= range(2,16)
for num_clust in n_cluster_agg:
    aggclustering = AgglomerativeClustering(n_clusters=num_clust, affinity = "euclidean", linkage="ward")
    agg_pred_train = aggclustering.fit_predict(X_train)
    agg_pred_test = aggclustering.fit_predict(X_test)
    
    silhouette_scores_train= silhouette_score(X_train,agg_pred_train) 
    silhouette_sccores_test = silhouette_score(X_test , agg_pred_test)
    train_pred.append(silhouette_scores_train)
    test_pred.append(silhouette_sccores_test)
cluster_num = list(range(2,16))
agg_df = pd.DataFrame({"Train Score":train_pred, "Test Score":test_pred}, index =cluster_num)
agg_df.index.name = "Cluster Num"
agg_df
