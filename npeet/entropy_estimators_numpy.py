import scipy.spatial as ss
from scipy.special import digamma
import numpy as np
import math


# CONTINUOUS ESTIMATORS

def entropy(x, k=3, base=math.e, noise_level=1e-10):
    """ The classic K-L k-nearest neighbor continuous entropy estimator.
    """
    assert k < x.shape[0]
    d = x.shape[1]
    n = x.shape[0]
    x += np.random.rand(*x.shape) * noise_level
    tree = ss.cKDTree(x)
    knn_distance = tree.query(x, [k + 1], p=float('inf'))[0]
    const = digamma(n) - digamma(k) + d * math.log(2)
    return (const + d * np.mean(np.log(knn_distance))) / math.log(base)


def conditional_entropy(x, y, k=3, base=math.e):
    """ The classic K-L k-nearest neighbor continuous entropy estimator for the
        entropy of X conditioned on Y.
    """
    assert x.shape[0] == y.shape[0]
    h_xy = entropy(np.concatenate((x, y), axis=-1), k, base)
    h_y = entropy(y, k, base)
    return h_xy - h_y


def mutual_information(x, y, k=3, base=math.e, noise_level=1e-10):
    """ Mutual information of x and y.
    """
    assert x.shape[0] == y.shape[0]
    assert k < x.shape[0]
    x += np.random.rand(*x.shape) * noise_level
    y += np.random.rand(*y.shape) * noise_level
    xy = np.concatenate((x, y), axis=-1)
    tree = ss.cKDTree(xy)
    knn_distance = tree.query(xy, [k + 1], p=float("inf"))[0]
    a, b, c, d = avg_digamma(x, knn_distance), avg_digamma(y, knn_distance), digamma(k), digamma(len(x))
    return (-a - b + c + d) / math.log(base)


def conditional_mutual_information(x, y, z, k=3, base=math.e, noise_level=1e-10):
    """ Mutual information of x and y, conditioned on z.
    """
    assert x.shape[0] == y.shape[0] == z.shape[0]
    assert k < x.shape[0]
    x += np.random.rand(*x.shape) * noise_level
    y += np.random.rand(*y.shape) * noise_level
    z += np.random.rand(*z.shape) * noise_level
    xyz = np.concatenate((x, y, z), axis=-1)
    xz = np.concatenate((x, z), axis=-1)
    yz = np.concatenate((y, z), axis=-1)
    tree = ss.cKDTree(xyz)
    knn_distance = tree.query(xyz, [k + 1], p=float("inf"))[0]
    a, b, c, d = avg_digamma(xz, knn_distance), avg_digamma(yz, knn_distance), avg_digamma(z, knn_distance), digamma(k)
    return (-a - b + c + d) / math.log(base)


def kl_divergence(x_p, x_q, k=3, base=math.e):
    """ KL Divergence between p and q for x_p~p(x), x_q~q(x).
    """
    assert k < x_p.shape[0] and k < x_q.shape[0]
    assert x_p.shape[1] == x_q.shape[1]
    d = x_p.shape[1]
    n = x_p.shape[0]
    m = x_q.shape[0]
    const = math.log(m) - math.log(n - 1)
    tree_p = ss.cKDTree(x_p)
    tree_q = ss.cKDTree(x_q)
    knn_distance_p = tree_p.query(x_p, [k + 1], p=float("inf"))[0]
    knn_distance_q = tree_q.query(x_q, [k], p=float("inf"))[0]
    return (const + d * np.log(knn_distance_q).mean() - d * np.log(knn_distance_p).mean()) / math.log(base)


def avg_digamma(x, knn_distance, eps=1e-15):
    n = x.shape[0]
    tree = ss.cKDTree(x)
    avg_arr = np.ndarray(shape=(n,))
    for index in range(n):
        num_points = len(tree.query_ball_point(x[index, :], knn_distance[index] - eps, p=float('inf')))
        avg_arr[index] = digamma(num_points)
    return avg_arr.mean()


if __name__ == "__main__":
    arr_1 = np.ones((100000, 1))
    arr_2 = np.random.normal(0, 1, (10000, 1))
    print(entropy(arr_2))
