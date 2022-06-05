'''
The reproduction code of the MicroAL module in 'BOOM-Explorer'
Author: Jialin Lu
'''

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def cal_distance(unsamples, centroids, k):
    # pre-defined diagonal weight matrix
    w = np.eye(4)
    w = w[0][0] * 10 # increase weight on features of interest, e.g. DecodeWidth

    dis = []
    for point in unsamples:
        dim = len(point)
        dis_temp = []
        for i in range(k):
            diff = np.array(point) - np.array(centroids[i])
            diff = diff.reshape(dim,1)
            diff_t = diff.T
            dis_i = np.dot(np.dot(diff_t,w),diff)[0][0]
            dis_temp.append(dis_i)
        dis.append(dis_temp)
    dis = np.array(dis)
    return dis

def update_cen(unsamples, centroids, k):
    distance = cal_distance(unsamples, centroids, k)
    minIndex = np.argmin(distance, axis=1)
    newCentroids = pd.DataFrame(unsamples).groupby(minIndex).mean()
    newCentroids = newCentroids.values

    changed = newCentroids - centroids
    return changed, newCentroids

def clustering(unsamples, k):
    '''
    unsamples: here is a list
    k: cluster nums
    return: centroids(list), cluster(list)
    '''
    centroids = random.sample(unsamples, k)
    changed, newCentroids = update_cen(unsamples, centroids, k)
    while np.any(changed):
        changed, newCentroids = update_cen(unsamples, newCentroids, k)
    centroids = sorted(newCentroids.tolist())

    cluster = []
    dis = cal_distance(unsamples, centroids, k)
    minIndex = np.argmin(dis, axis=1)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minIndex):
        cluster[j].append(unsamples[i])
    return centroids, cluster

def ted(unsamples:list, sample_nums, normalization_mu=1e-3):
    '''
    unsamples: (m*n), m is unsample nums, n is samples' dim.
    sample_nums: sample nums to be selected from unsamples.
    return: (sample_nums*n) samples selected from unsamples.
    '''
    unsamples = np.array(unsamples)
    samples = []
    K = rbf_kernel(X=unsamples, Y=unsamples)
    for i in range(sample_nums):
        scores = np.zeros((unsamples.shape[0], 1))
        for j in range(unsamples.shape[0]):
            score_tmp = np.dot(K[j,:].reshape(1,-1), K[:,j].reshape(-1,1)) / (K[j,j] + normalization_mu)
            scores[j] = score_tmp
        idx = scores.argmax()
        samples.append(list(unsamples[idx]))
        
        # update unsamples and K
        unsamples = np.delete(unsamples, idx, axis=0)
        K = K - np.dot(K[:,idx].reshape(-1,1), K[idx,:].reshape(1,-1)) / (K[idx,idx] + normalization_mu)
    
    return samples

def microAL(unsamples:list, total_sample_nums:int, cluster_nums:int) -> list:
    '''
    unsamples: (m*n), m is unsample nums, n is samples' dim, here is a list.
    total_sample_nums: sample nums to be selected from unsamples.
    cluster_nums: depends on the variable space of the feature of interest
    return: (sample_nums*n) micro samples selected from unsamples.
    '''
    micro_samples = []

    centroids, clusters = clustering(unsamples, cluster_nums)
    for cluster in clusters:
        cluster_samples = ted(cluster, round(total_sample_nums/cluster_nums))
        for sample in cluster_samples:
            micro_samples.append(sample)

    return micro_samples

def kmeans_test(data):
    k = 4
    model = KMeans(n_clusters=k)
    model.fit(data)
    centers = model.cluster_centers_
    result = model.predict(data)
    mark = ['or', 'og', 'ob', 'ok']
    for i, d in enumerate(data):
        plt.plot(d[0], d[1], mark[result[i]])
    mark = ['*r', '*g', '*b', '*k']
    for i, center in enumerate(centers):
        plt.plot(center[0], center[1], mark[i], markersize=20)

    #plt.show()
    plt.savefig('kmeans.png')

def my_clustering_test(data):
    centroids, clusters = clustering(list(data), 4)
    mark = ['or', 'og', 'ob', 'ok']
    for i, cluster in enumerate(clusters):
        for d in cluster:
            plt.plot(d[0], d[1], mark[i])
    mark = ['*r', '*g', '*b', '*k']
    for i, center in enumerate(centroids):
        plt.plot(center[0], center[1], mark[i], markersize=20)
    
    # microAL test
    micro_samples = microAL(list(data), 12, 4)
    for sample in micro_samples:
        plt.plot(sample[0], sample[1], marker='$\circledS$', color='purple', markersize=18)

    #plt.show()
    plt.savefig('my_clustering.png')

if __name__ == '__main__':
    # load data
    test_data = np.genfromtxt('test_data.txt', delimiter=' ')
    # run k-means 
    kmeans_test(test_data)
    # run my clustering
    my_clustering_test(test_data)

