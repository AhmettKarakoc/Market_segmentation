def warn (*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.impute import KNNImputer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans,DBSCAN
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from matplotlib.patches import Ellipse
%matplotlib inline

df = pd.read_csv("Customerdata.csv")
df.sample(5)

df.shape
df.info()
df.describe().T
df.isnull().sum()
null_columns = df.columns[df.isnull().any()]
df[null_columns].isnull().sum()
#In this code snippet, I used the KNNImputer to fill missing values using the K-Nearest Neighbors (KNN) method.
clean = KNNImputer(n_neighbors=7,weights="distance")
numerical = df[null_columns].select_dtypes(exclude = "object").columns
df[numerical] = clean.fit_transform(df[numerical])
df.drop(columns=['CUST_ID'], inplace=True)