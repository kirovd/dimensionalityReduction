#this is to load the data
import numpy as np

#this is to display the data
import pandas as pd

#this is for making the the graph/heatmap visualisation
import seaborn as sns

#METHODS#

#This is the method used for data reduction as its one of the most popular
from sklearn.decomposition import PCA

#this is the second method used for data reduction
from sklearn.manifold import MDS

#this is the third method used for data reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


#CLUSTERING#

#this is to do any visual plots using scatter plots
import matplotlib.pyplot as plt

#this is to do clustering
from sklearn.cluster import KMeans


#NORMALIZATIONS#

#using StandardScaler is the most common preprocessing tool for any dataset
from sklearn.preprocessing import StandardScaler


#UMAP VISUALISATION#

#in order for umap to run it needs the following imports
import umap
import umap.plot
import datashader as ds
import datashader.transfer_functions as tf
import datashader.bundling as bd
import colorcet
import matplotlib.colors
import matplotlib.cm
import bokeh.plotting as bpl
import bokeh.transform as btr
import holoviews as hv
import holoviews.operation.datashader as hd

X = np.load("C:/Users/dekii/genesdata/X.npy")
Y = np.load("C:/Users/dekii/genesdata/y.npy")
df = pd.DataFrame(X)
print('Shape of Data = ', X.shape)
print('No. of Samples = ', X.shape[0])
print('No. of Genes = ', X.shape[1])

df[[0,1,2,3,4,5,10,100,1000,10000,20000,40000,45000]].describe()

sns.displot(data=df[[0,1,2,3,4,5]], kind='hist',height=5)

scalar = StandardScaler()
scaled_data = pd.DataFrame(scalar.fit_transform(X)) 
scaled_data

sns.displot(data=scaled_data[[0,1,2,3]], kind='hist',height=7)

pca = PCA()
x_pca = pca.fit_transform(scaled_data)
print('Shape of Data Projected on PCA = ', x_pca.shape)

pca_normalized = PCA()
x_pca_normalized = pca_normalized.fit_transform(scaled_data)

fig, ax = plt.subplots(2,1,figsize=(12,18))
ax[0].scatter(x_pca[:,0],x_pca[:,1])
ax[0].set_title('Gene Scatter Plot - 2 PCA')
ax[0].set_xlabel('PCA - 1')
ax[0].set_ylabel('PCA - 2')

ax[1].scatter(x_pca_normalized[:,0],x_pca_normalized[:,1])
ax[1].set_title('Gene (Log) Scatter Plot - 2 PCA')
ax[1].set_xlabel('PCA - 1')
ax[1].set_ylabel('PCA - 2')

plt.show()

pca = PCA(n_components = 4)
pca.fit(scaled_data)
data_pca = pca.transform(scaled_data)
data_pca = pd.DataFrame(data_pca,columns=['PC1','PC2','PC3','PC4'])
data_pca.head()

sns.displot(data=data_pca, kind='hist',height=5)

pca = PCA(n_components=4) # PCs used
x_pca4 = pca.fit_transform(x_pca_normalized)

pca_4 = PCA(n_components=4)
x_pca_4 = pca_4.fit_transform(x_pca4)

number_of_cluster = 4
kmeans_pca = KMeans(n_clusters=number_of_cluster)
x_kmeans_pca = kmeans_pca.fit_transform(x_pca_4)

fig, ax = plt.subplots(3,1,figsize=(12,18))
ax[0].scatter(x_pca4[:,0],x_pca4[:,1])
ax[0].set_title('Transformed (Log) Data in Lower Dimension (4)')
ax[0].set_xlabel('Dimension - 1')
ax[0].set_ylabel('Dimension - 2')

ax[1].scatter(x_pca_4[:,0],x_pca_4[:,1])
ax[1].set_title('Transformed (Log) Data in Lower Dimension (4) - 2 PCAs')
ax[1].set_xlabel('PCA - 1')
ax[1].set_ylabel('PCA - 2')

ax[2].scatter(x_pca_4[:,0],x_pca_4[:,1], c =kmeans_pca.labels_)
ax[2].set_title('Transformed (Log) Data in Lower Dimension (4) - 3 Clusters')
ax[2].set_xlabel('PCA - 1')
ax[2].set_ylabel('PCA - 2')

mapping_plot = umap.UMAP().fit(x_pca_4)

umap.plot.connectivity(mapping_plot, show_points=True)

mds = MDS()
x_mds = mds.fit_transform(scaled_data)
print('Shape of Data Projected on MDS = ', x_mds.shape)

mds_normalized = MDS()
x_mds_normalized = mds_normalized.fit_transform(scaled_data)

fig, ax = plt.subplots(2,1,figsize=(12,18))
ax[0].scatter(x_mds[:,0],x_mds[:,1])
ax[0].set_title('Gene (Log) Scatter Plot - MDS 2D')
ax[0].set_xlabel('MDS - 1')
ax[0].set_ylabel('MDS - 2')

ax[1].scatter(x_mds_normalized[:,0],x_mds_normalized[:,1])
ax[1].set_title('Gene (Log) Scatter Plot - MDS 2D')
ax[1].set_xlabel('MDS - 1')
ax[1].set_ylabel('MDS - 2')

plt.show()

mds = MDS(n_components=4)
x_mds4 = mds.fit_transform(x_mds_normalized)

mds4 = MDS(n_components=4)
x_mds_4 = mds4.fit_transform(x_mds4)

number_of_cluster = 4
kmeans_mds = KMeans(n_clusters=number_of_cluster)
x_kmeans_mds = kmeans_mds.fit_transform(x_mds_4)

fig, ax = plt.subplots(3,1,figsize=(12,18))
ax[0].scatter(x_mds4[:,0],x_mds4[:,1])
ax[0].set_title('Transformed (Log) Data in Lower Dimension (4)')
ax[0].set_xlabel('Dimension - 1')
ax[0].set_ylabel('Dimension - 2')

ax[1].scatter(x_mds_4[:,0],x_mds_4[:,1])
ax[1].set_title('Transformed (Log) Data in Lower Dimension (4) - MDS 2D')
ax[1].set_xlabel('MDS - 1')
ax[1].set_ylabel('MDS - 2')

ax[2].scatter(x_mds_4[:,0],x_mds_4[:,1], c =kmeans_mds.labels_)
ax[2].set_title('Transformed (Log) Data in Lower Dimension (4) - MDS 2D in Cluster')
ax[2].set_xlabel('MDS - 1')
ax[2].set_ylabel('MDS - 2')

mapping_plot_mds = umap.UMAP().fit(x_mds_4)

umap.plot.connectivity(mapping_plot_mds, show_points=True)

lda = LDA(n_components=4)
x_lda = lda.fit_transform(scaled_data,Y)

lda_normalized = LDA(n_components=4)
x_lda_normalized = lda_normalized.fit_transform(x_lda, Y)

fig, ax = plt.subplots(2,1,figsize=(10,14))
ax[0].scatter(x_lda[:,0],x_lda[:,1])
ax[0].set_title('Gene (Log) Scatter Plot - LDA 2D')
ax[0].set_xlabel('LDA - 1')
ax[0].set_ylabel('LDA - 2')

ax[1].scatter(x_lda_normalized[:,0],x_lda_normalized[:,1])
ax[1].set_title('Gene (Log) Scatter Plot - LDA 2D')
ax[1].set_xlabel('LDA - 1')
ax[1].set_ylabel('LDA - 2')

plt.show()

lda = LDA(n_components=4)
x_lda4 = lda.fit_transform(x_lda_normalized,Y)

lda4 = LDA(n_components=4)
x_lda_4 = lda4.fit_transform(x_lda4,Y)

number_of_cluster = 4
kmeans_lda = KMeans(n_clusters=number_of_cluster)
x_kmeans_lda = kmeans_lda.fit_transform(x_lda_4)

fig, ax = plt.subplots(3,1,figsize=(12,18))
ax[0].scatter(x_lda4[:,0], x_lda4[:,1])
ax[0].set_title('Transformed (Log) Data in Lower Dimension (4)')
ax[0].set_xlabel('Dimension - 1')
ax[0].set_ylabel('Dimension - 2')

ax[1].scatter(x_lda_4[:,0],x_lda_4[:,1])
ax[1].set_title('Transformed (Log) Data in Lower Dimension (4) - LDA 2D')
ax[1].set_xlabel('LDA - 1')
ax[1].set_ylabel('LDA - 2')

ax[2].scatter(x_lda_4[:,0],x_lda_4[:,1], c =kmeans_lda.labels_)
ax[2].set_title('Transformed (Log) Data in Lower Dimension (4) - LDA 2D in Cluster')
ax[2].set_xlabel('LDA - 1')
ax[2].set_ylabel('LDA - 2')

mapping_plot_lda = umap.UMAP().fit(x_lda_4)

umap.plot.connectivity(mapping_plot_lda, show_points=True)

fig, ax = plt.subplots(2,2,figsize=(18,12))

ax[0,0].scatter(X[:,0], x_pca_4[:,1])
ax[0,0].set_title('Genes data Plot - Genes vs PCA 2D')
ax[0,0].set_xlabel('Genes vs PCA - 1')
ax[0,0].set_ylabel('Genes vs PCA - 2')

ax[0,1].scatter(x_pca_4[:,0],x_mds_4[:,1])
ax[0,1].set_title('Gene Scatter Plot - PCA vs MDS 2D')
ax[0,1].set_xlabel('PCA vs MDS - 1')
ax[0,1].set_ylabel('PCA vs MDS - 2')

ax[1,0].scatter(x_mds_4[:,0],x_lda_4[:,1])
ax[1,0].set_title('Gene Scatter Plot - MDS vs LDA 2D')
ax[1,0].set_xlabel('MDS vs LDA - 1')
ax[1,0].set_ylabel('MDS vs LDA - 2')

ax[1,1].scatter(x_lda_4[:,0],x_pca_4[:,1])
ax[1,1].set_title('Gene Scatter Plot - LDA vs PCA 2D')
ax[1,1].set_xlabel('LDA vs PCA - 1')
ax[1,1].set_ylabel('LDA vs PCA - 2')

plt.show()