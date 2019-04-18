from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
import random
import math
import time

color_dict = defaultdict()


def init_color_dict(number_of_clusters):
    color_dict[0] = '#448aff'
    color_dict[1] = '#ec407a'
    color_dict[2] = '#00e676'
    color_dict[3] = '#ff6f00'
    color_dict[4] = '#ef9a9a'
    for i in range(5, number_of_clusters):
        color_dict[i] = np.random.rand(1, 3)


# for testing only - generate n normalized around 3 centers
def generate_2d_dataset_with_3_centers(n):
    res = []
    sigma = [[0.1, 0.25], [0.5, 0.1], [0.2, 0.4]]
    mu = [[0.5, 1.5], [2, 0.5], [3.5, 2.5]]
    for i in range(n):
        x = np.random.standard_normal() * sigma[i % 3][0] + mu[i % 3][0]
        y = np.random.standard_normal() * sigma[i % 3][1] + mu[i % 3][1]
        res.append([x, y])
    return res


# generate n data each with d features
def generate_dataset(n, d, data_min, data_max):
    res = []
    for i in range(n):
        sub = []
        for j in range(d):
            sub.append(random.uniform(data_min, data_max))
        res.append(sub)
    return res


def print_dataset(dataset):
    print(np.asarray(dataset))


def plot_2d_dataset(dataset, markers=None, labels=None):
    # x, y = np.asarray(dataset).T
    if labels is None:
        labels = []
    if markers is None:
        markers = []

    for i in range(len(dataset)):
        point = dataset[i]
        plt.scatter(point[0], point[1], c=color_dict[labels[i]])

    for marker in markers:
        plt.scatter(marker[0], marker[1], s=100, c=color_dict[markers.index(marker)], marker='+')
    plt.show()
    # plt.savefig(str(time.time()) + '.png')
    # plt.close()


def plot_2d_with_markers(clusters, dataset, membership):
    labels = get_cluster_labels(membership)
    plot_2d_dataset(dataset, markers=clusters, labels=labels)


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
        sigma_u_ik_m = sum(u_ik_m)  # i is fixed
        weighted_data = []
        for k in range(len(dataset)):
            weighted_vector = []
            for f in range(len(dataset[k])):  # iterate over features
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
    membership_matrix = []
    fuzzy_power = float(2 / (m-1))
    n = len(dataset)
    c = len(clusters)
    for i in range(n):
        denom = sum([(1/euclidean_distance(dataset[i], clusters[x])) ** fuzzy_power for x in range(c)])
        membership = []
        for j in range(c):
            num = (1/euclidean_distance(dataset[i], clusters[j])) ** fuzzy_power
            membership.append(num/denom)
        membership_matrix.append(membership)
    return membership_matrix


def get_cluster_labels(membership_matrix):
    res = []
    for membership in membership_matrix:
        max_index = membership.index(max(membership))
        res.append(max_index)
    return res


def fcm(cluster_no, iterations, dataset, m=2):
    c = cluster_no
    n = len(dataset)
    membership = generate_random_membership_function(n, c)
    clusters = []
    for i in range(iterations):
        clusters = update_cluster_centers(dataset, membership, m)
        membership = update_membership_matrix(dataset, clusters, m)
        # plot_2d_with_markers(clusters, dataset, membership) # this is only working if dataset is 2d
    return clusters, membership


def run_fcm_on_a_2d_dataset():
    features = 2
    number_of_data = 100
    number_of_clusters = 3
    iterations = 10
    data_max = 3
    data_min = 0
    dataset = generate_dataset(number_of_data, features, data_min, data_max)  # generate a random dataset
    # dataset = generate_2d_dataset_with_3_centers(number_of_data)
    init_color_dict(number_of_clusters)  # init colors for plotting
    cluster_centers, final_memberships = fcm(number_of_clusters, iterations, dataset)  # run FCM
    plot_2d_with_markers(cluster_centers, dataset, final_memberships)  # plot final result
    final_labels = get_cluster_labels(final_memberships)  # get labels
    print("cluster centers:")
    print_dataset(cluster_centers)


if __name__ == '__main__':
    run_fcm_on_a_2d_dataset()
