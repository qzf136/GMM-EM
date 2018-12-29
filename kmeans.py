import numpy as np
import matplotlib.pyplot as plt
import random


def generate_data():
    X = []
    k = 4
    number = 20
    p = [0.5, 0.5, 0.5, 0.5]
    u = [[4,5], [6,4], [4,3], [6,1]]
    sigma = [[0.5,0.5], [0.5,0.5], [0.5,0.5], [0.5,0.5]]
    for i in range(k):
        cov = [[sigma[i][0]**2, sigma[i][0]*sigma[i][1]*p[i]],
               [sigma[i][0]*sigma[i][1]*p[i], sigma[i][1]**2]]
        x = np.random.multivariate_normal(u[i], cov, number)
        x = x.tolist()
        X = X + x
    return X, k


def init_center(X, k):
    x_center = random.sample(X, k)
    return x_center


def calculate_distance(x1, x2):
    sum = 0
    n = len(x1)
    for i in range(n):
        sum += (x1[i]-x2[i]) ** 2
    return sum


def closest(x, x_center):
    dis = []
    for i in range(len(x_center)):
        dis.append(calculate_distance(x, x_center[i]))
    min_dis = min(dis)
    index = dis.index(min_dis)
    return x_center[index]


def min_distance(X, x_center):
    dict = {}
    for x_c in x_center:
        dict[tuple(x_c)] = []
    for x_sample in X:
        center = closest(x_sample, x_center)
        dict[tuple(center)].append(x_sample)
    return dict


def get_centroid(X, dict):
    x_center = []
    for key in dict.keys():
        lst = dict[key]
        center = np.mean(lst, axis=0)
        x_center.append(center)
    return x_center


def get_sum_dis(dict):
    sum = 0
    for key in dict.keys():
        xs = dict[key]
        for x in xs:
            dis = calculate_distance(x, list(key))
            sum += dis
    return sum


def kmeans(X, k):
    x_center = init_center(X, k)
    dict = min_distance(X, x_center)
    new_value = get_sum_dis(dict)
    old_value = 0
    while (abs(new_value - old_value) > 1):
        x_center = get_centroid(X, dict)
        dict = min_distance(X, x_center)
        old_value = new_value
        new_value = get_sum_dis(dict)
    return dict, new_value


def repeat_kmeans(X, k):
    min_val = 1e4
    dict = {}
    for iter in range(10):
        dic, val = kmeans(X, k)
        if val < min_val:
            min_val = val
            dict = dic
    return dict, min_val


def show(dict):
    for key in dict:
        lst = dict[key]
        arr = np.array(lst)
        x1 = arr[:,0]
        x2 = arr[:,1]
        plt.scatter(x1, x2)
    plt.show()


X, k = generate_data()
dict, min_val = repeat_kmeans(X, k)
show(dict)
