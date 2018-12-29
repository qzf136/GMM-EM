import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy

# https://archive.ics.uci.edu/ml/datasets/seeds
def read_data(file_name):
    file = open("data.txt")
    X = []
    for line in file:
        a = line.split()
        b = a[0:len(a)-1]
        c = list(map(float, b))
        X.append(c)
    return X

def generate_data():
    X = []
    k = 4
    number = 20
    p = [0.5, 0.5, 0.5, 0.5]
    u = [[3,3], [3,7], [7,3], [7,7]]
    sigma = [[0.5,0.5], [0.5,0.5], [0.5,0.5], [0.5,0.5]]
    for i in range(k):
        cov = [[sigma[i][0]**2, sigma[i][0]*sigma[i][1]*p[i]],
               [sigma[i][0]*sigma[i][1]*p[i], sigma[i][1]**2]]
        x = np.random.multivariate_normal(u[i], cov, number)
        x = x.tolist()
        X = X + x
    return X, k


def init_parameter(X, k):
    pi = []
    for i in range(k-1):
        pi.append(1/k)
    pi.append(1-sum(pi))
    mean = random.sample(X, k)
    cov = [np.eye(np.shape(X)[1])] * k
    return pi, mean, cov


def gauss(x, mean, cov):
    dim = np.shape(cov)[0]
    cov = np.mat(cov)
    cov_det = np.linalg.det(cov)
    if cov_det == 0:
        cov = cov + np.eye(dim) * 0.01
        cov_det = np.linalg.det(cov)
    val_1 = 1 / (pow(pow(2*np.pi, dim) * abs(cov_det), 0.5))
    val_2 = np.exp(-0.5 * (np.mat(x) - np.mat(mean)) * cov.I * (np.mat(x) - np.mat(mean)).T)[0,0]
    if val_1 * val_2 == 0:
        return 0.000001
    else:
        return val_1*val_2


def E(X, pi, mean, cov):
    k_num = len(pi)
    grama = []
    for n in range(len(X)):
        sum = 0
        lst = []
        for i in range(k_num):
            p = gauss(X[n], mean[i], cov[i])
            sum += pi[i] * p
        for k in range(k_num):
            lst.append(pi[k] * gauss(X[n], mean[k], cov[k]) / sum)
        grama.append(lst)
    return grama


def M(X, mean, grama):
    grama = np.array(grama)
    k_num = np.shape(grama)[1]
    mean_new = []
    pi_new = []
    cov_new = []
    for k in range(k_num):
        Nk = sum(grama[:,k])
        meank = np.array(grama[:,k].T * np.mat(X) / Nk)[0,:]
        v2 = [((np.mat(x)-np.mat(mean[k])).T * (np.mat(x)-np.mat(mean[k]))) for x in X]
        covk = sum([grama[i,k] * v2[i] for i in range(len(X))]) / Nk
        pik = Nk / len(X)
        pi_new.append(pik)
        mean_new.append(meank)
        cov_new.append(covk)
    return pi_new, mean_new, cov_new


def calculate_likehood(X, pi, mean, cov):
    sum_N = 0
    for n in range(len(X)):
        sum_K = 0
        for k in range(len(pi)):
            # print(pi[k])
            # print(X[n])
            # print(mean[k])
            # print(cov[k])
            val = pi[k] * gauss(X[n], mean[k], cov[k])
            sum_K += val
        # print(sum_K)
        sum_N += math.log(sum_K)
    return sum_N


def EM(X, k):
    pi, mean, cov = init_parameter(X, k)
    old_val = 0
    new_val = calculate_likehood(X, pi, mean, cov)
    while abs(old_val - new_val) > 1:
        grama = E(X, pi, mean, cov)
        pi, mean, cov = M(X, mean, grama)
        old_val = new_val
        new_val = calculate_likehood(X, pi, mean, cov)
    return pi, mean, cov


def dict(X, pi, mean, cov):
    dic = {}
    for i in range(len(pi)):
        dic[i] = []
    grama = E(X, pi, mean, cov)
    for i in range(len(X)):
        lst = grama[i]
        max_p = max(lst)
        index = lst.index(max_p)
        dic[index].append(X[i])
    return dic


def show(dic):
    for key in dic:
        lst = dic[key]
        arr = np.array(lst)
        x1 = arr[:,0]
        x2 = arr[:,1]
        plt.scatter(x1, x2)
    plt.show()


def repeat(X, k):
    max_val = -10000
    pi_final = []
    mean_final = []
    cov_final = []
    for i in range(5):
        pi, mean, cov = EM(X, k)
        val = calculate_likehood(X, pi, mean, cov)
        if val > max_val:
            max_val = val
            pi_final = copy.copy(pi)
            mean_final = copy.copy(mean)
            cov_final = copy.copy(cov)
    return pi_final, mean_final, cov_final


X, k = generate_data()
pi, mean, cov = repeat(X, k)
print("pi = ", pi)
print("mean = ", mean)
print("cov = ", cov)
dic = dict(X, pi, mean, cov)
show(dic)


# X = read_data("data.txt")
# k = 3
# pi, mean, cov = repeat(X, k)
# print("pi = ", pi)
# print("mean = ", mean)
# print("cov = ", cov)
# dic = dict(X, pi, mean, cov)



