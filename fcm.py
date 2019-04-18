import numpy as np
from matplotlib import pyplot as plt
import random
import math


# generate n data each with d features
def generate_dataset(n, d, min, max):
    res = []
    for i in range(n):
        sub = []
        for j in range(d):
            sub.append(random.uniform(min, max))
        res.append(sub)
    return res


def print_dataset(dataset):
    print(np.asarray(dataset))


def plot_2d_dataset(dataset, markers=[]):
    x, y = np.asarray(dataset).T
    plt.scatter(x, y)
    for marker in markers:
        plt.scatter(marker[0], marker[1], s=100, c='red', marker='+')
    plt.show()


# return a n * c membership function
# n = number of data
# c = number of clusters
def generate_random_membership_function(n, c):
    membership = np.random.rand(n, c)
    summation = [sum(center) for center in membership]
    normalized = []
    for i in range(len(membership)):
        tmp = []
        for d in membership[i]:
            tmp.append(d / summation[i])
        normalized.append(tmp)
    return normalized

# calculate weighted average of data points
# for each cluster i calculate sigma[k from 1 to n]((U_ik)^m * DATA_k)/sigma[k from 1 to n]((U_ik)^m)
# m is fuzziness parameter
def update_cluster_centers(dataset, membership_matrix, m):
    number_of_clusters = len(membership_matrix[0])
    cluster_centers = []
    for i in range(number_of_clusters):
        u_ik = list(zip(*membership_matrix))[i]
        u_ik_m = [x ** m for x in u_ik]
        sigma_u_ik_m = sum(u_ik_m) # i is fixed
        weighted_data = []
        for k in range(len(dataset)):
            weighted_vector = []
            for f in range(len(dataset[k])): # iterate over features
                weighted_vector.append(u_ik_m[k] * dataset[k][f])
            weighted_data.append(weighted_vector)
        sigma_data_u_ik_m = [sum(x) for x in list(zip(*weighted_data))]
        cluster_centers.append([sigma_data_u_ik_m[d]/sigma_u_ik_m for d in range(len(sigma_data_u_ik_m))])
    return cluster_centers


def euclidean_distance(p, q):
    summation = 0
    for i in range(len(p)):
        summation += (p[i] - q[i]) ** 2
    return math.sqrt(summation)


def update_membership_matrix(dataset, clusters, m):
    raise NotImplementedError


c = 2
f = 2
n = 3
m = 2
dataset = generate_dataset(n, f, 0, 3)
print("dataset:")
print_dataset(dataset)
plot_2d_dataset(dataset)
print("membership:")
membership = generate_random_membership_function(n, c)
print_dataset(membership)
cc = update_cluster_centers(dataset, membership,m)
print("cluster centers:")
print_dataset(cc)