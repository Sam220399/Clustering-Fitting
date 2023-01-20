# The following import statements import all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import sklearn.metrics as skmet
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
import sklearn.cluster as cluster

# Scaling the data using MinMaxScaler, to nullify the effect of high magnitude values in a particular column.
def scale(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)

# This function plots inertia vs number of clusters, so that we can determine the optimal number of clusters. 
# i.e. k value in k-means clustering
def elbow_plot(data):
    data = list(zip(data[:, 0], data[:, 1]))
    inertias = []
    for x in range(1,11):
        kmeans = cluster.KMeans(n_clusters=x)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    plt.plot(range(1,11), inertias, marker='.')
    plt.title('Elbow plot')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

# Finding the index of first country which belongs to cluster n.
def find_index(clusters, n):
    for i in range(len(clusters)):
        if (clusters[i] == n):
            return i

# This function determines the relation ship between x and y values of the courve y=f(x).
def objective(x, a, b):
    return a*x + b

# Trying to come up with a curve which fits the data.
def fit_curve(X, Y):
    popt, _ = curve_fit(objective, X, Y)
    a, b = popt
    plt.scatter(X, Y)
    plt.xlabel('Year')
    plt.ylabel('Forest area in Sq.km.')
    plt.title('Forest area in Sq.km in Arab World')
    x_line = np.arange(min(X), max(X), 1)
    y_line = objective(x_line, a, b)
    plt.plot(x_line, y_line, '--', color='red')
    plt.show()
    # Predicting the forest area from the curve.
    forest_area = [2055, 2060, 2070, 2080]
    for i in range(len(forest_area)):
        print('The predicted forest area in Arab World in', forest_area[i], 'is', objective(forest_area[i], a, b))

# Plotting k number of clusters using k-means clustering algorithm.
def plot_kmeans(data, n):
    kmeans = cluster.KMeans(n_clusters = n)
    kmeans.fit(data)
    cen = kmeans.cluster_centers_
    # Printing the centers of the clusters.
    for i in range(n):
        print('The coordinates of the center of cluster', i+1, 'are (', cen[i, 0], ',', cen[i, 1], ')')
    # Printing the silhouette score.
    print('The Silhouette score of the clusters is ', skmet.silhouette_score(data, kmeans.labels_))

    # Plotting the scaled clusters.
    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_)
    plt.xlabel('Scaled values of the amount of CO2 emitted')
    plt.ylabel('Scaled values of the forest area')
    plt.title('K-means clustering')
    for i in range(n):
        plt.plot(cen[i, 0], cen[i, 1], "*", markersize=10, c='r')
    plt.show()
    return kmeans

# Comparing samples of different clusters.
def compare_the_countries(data, first_country, second_country):
    print(data.iloc[[first_country, second_country], :])

# The following lines of code will load the data in csv files into a data frame object
co2_data = pd.read_csv('co2_emission_data.csv')
forest_data = pd.read_csv('forest_area.csv')

# Combining the above two dataframe objects into a single single dataframe object
data = pd.DataFrame({'Country' : co2_data.iloc[:, 0].values,
                    'Amount of CO2 emitted' : co2_data.iloc[:, 63],
                    'Forest Area' : forest_data.iloc[:, 63]})

# Handling the missing values. As there are only a few rows with missing values, it is better to drop them.
data = data.dropna()
data_before_scaling = data

# Handling the difference in units of different columns. We can address this issue by normalization.
data = scale(data.iloc[:, 1:])

# Determining the value of k in k-means clustering using the elbow method.
elbow_plot(data)

# Using k-means clustering algorithm to determine and plot the clusters.
kmeans = plot_kmeans(data, 2)

# Comparing two countries data belonging to two different clusters
country_from_first_cluster = find_index(kmeans.labels_, 0)
country_from_second_cluster = find_index(kmeans.labels_, 1)
compare_the_countries(data_before_scaling, country_from_first_cluster, country_from_second_cluster)

# Fitting a curve.
fit_curve(range(1990, 2011), forest_data.iloc[7, 34:55])