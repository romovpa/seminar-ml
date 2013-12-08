#!/usr/bin/env python
"""
Simple implementations of SVD-based collaborative filtering algorithms.
"""

import sys
import numpy as np


# TODO: implement bayesian SVD
# TODO: implement ALS1


def svd_als(data, preferences=1.0, confidences=1.0, rank=10, penalty=0.1, p_default=0.0, c_default=0.0, iters=10,
            init_X=None, init_Y=None, verbose=False):
    """
    Trains low-rank matrix factorization model using alternative least square algorithm.

    Prediction model:
        r_ij = x_i * y_j

    Training objective:
        Sum_ij { c_ij (p_ij - x_i * y_j)^2 } + lambda ( Sum_i ||x_i||^2 + Sum_j ||y_j||^2 )

    Args
        data: pair (users, items) of numpy integer vectors
        rank: rank of SVD decomposition
        penalty: regularization coefficient (lambda)
        iters: number of ALS iterations to perform
        init_X:
        init_Y: initial values of factors (normal random sample by default)
        p_default:
        c_default:

    References:
        - Hu et al. 2008 Collaborative Filtering for Implicit Feedback Datasets
    """

    if verbose:
        print 'Preparing the data...'

    users, items = data

    n_users = users.max()+1
    n_items = items.max()+1
    n_points = len(users)

    if np.isscalar(preferences):
        preferences = preferences * np.ones(n_points)
    if np.isscalar(confidences):
        confidences = confidences * np.ones(n_points)

    user_order = users.argsort()
    item_order = items.argsort()
    user_ordered_users = np.copy(users[user_order])
    user_ordered_items = np.copy(items[user_order])
    user_ordered_preferences = np.copy(preferences[user_order])
    user_ordered_confidences = np.copy(confidences[user_order])
    item_ordered_users = np.copy(users[item_order])
    item_ordered_items = np.copy(items[item_order])
    item_ordered_preferences = np.copy(preferences[item_order])
    item_ordered_confidences = np.copy(confidences[item_order])
    user_items = [None] * n_users
    user_preferences = [None] * n_users
    user_confidences = [None] * n_users
    item_users = [None] * n_items
    item_preferences = [None] * n_items
    item_confidences = [None] * n_items
    ptr = 0
    for i in xrange(1, n_points+1):
        if i == n_points or user_ordered_users[i] != user_ordered_users[ptr]:
            user = user_ordered_users[ptr]
            user_items[user] = user_ordered_items[ptr:i]
            user_preferences[user] = user_ordered_preferences[ptr:i]
            user_confidences[user] = user_ordered_confidences[ptr:i]
            ptr = i
    ptr = 0
    for i in xrange(1, n_points+1):
        if i == n_points or item_ordered_items[i] != item_ordered_items[ptr]:
            item = item_ordered_items[ptr]
            item_users[item] = item_ordered_users[ptr:i]
            item_preferences[item] = item_ordered_preferences[ptr:i]
            item_confidences[item] = item_ordered_confidences[ptr:i]
            ptr = i

    if init_X is None:
        init_X = np.random.standard_normal((n_users, rank)) / rank
    if init_Y is None:
        init_Y = np.random.standard_normal((n_items, rank)) / rank

    X = init_X
    Y = init_Y

    if verbose:
        print 'Starting iterations...'

    for iter in xrange(iters):

        # Update X
        V0 = np.dot(Y.transpose(), Y) + penalty*np.eye(rank)
        b0 = p_default * c_default * Y.transpose().sum(axis=1)
        for user in xrange(n_users):
            rel_items = user_items[user]
            rel_preferences = user_preferences[user]
            rel_confidences = user_confidences[user]
            if rel_preferences is None:
                continue
            rel_count = len(rel_items)

            V1u = np.dot(Y[rel_items,:].transpose(),
                         (rel_confidences - c_default).reshape((rel_count, 1))*Y[rel_items,:])
            Vu = V0 + V1u
            b1u = np.dot(Y[rel_items,:].transpose(), rel_confidences * (rel_preferences - p_default) +
                p_default * (rel_confidences - c_default))
            b = b0 + b1u
            x = np.linalg.solve(Vu, b)
            X[user,:] = x.transpose()

        # Update Y
        V0 = np.dot(X.transpose(), X) + penalty*np.eye(rank)
        b0 = p_default * c_default * X.transpose().sum(axis=1)
        for item in xrange(n_items):
            rel_users = item_users[item]
            rel_preferences = item_preferences[item]
            rel_confidences = item_confidences[item]
            if rel_preferences is None:
                continue
            rel_count = len(rel_users)

            V1i = np.dot(X[rel_users,:].transpose(),
                         (rel_confidences - c_default).reshape((rel_count, 1))*X[rel_users,:])
            Vi = V0 + V1i
            b1u = np.dot(X[rel_users,:].transpose(), rel_confidences * (rel_preferences - p_default) +
                p_default * (rel_confidences - c_default))
            b = b0 + b1u
            y = np.linalg.solve(Vi, b)
            Y[item,:] = y.transpose()

        if verbose:
            # Compute RMSE
            predictions = (X[users,:]*Y[items,:]).sum(axis=1)
            rmseR = np.sqrt(np.square(preferences - predictions).mean())

            # Compute objective value
            J = penalty*(np.square(X).sum() + np.square(Y).sum())
            for user in xrange(n_users):
                rel_items = user_items[user]
                rel_preferences = user_preferences[user]
                rel_confidences = user_confidences[user]
                if rel_preferences is None:
                    continue
                rel_count = len(rel_items)

                predictions = np.dot(Y, X[user,:].transpose())
                rel_predictions = predictions[rel_items]
                other_predictions = predictions - p_default
                other_predictions[rel_items] -= rel_predictions - p_default
                J += (np.dot(np.square(rel_preferences - rel_predictions), rel_confidences) +
                      c_default * np.square(other_predictions).sum())

            # Information
            print 'Iter %d: J=%.2f RMSE(R)=%.8f' % (iter, J, rmseR)

    return X, Y


def shift(data, values, reverse=False, iters=1):
    """
    Finds optimal shift with ALS.
    """
    users, items = data

    n_users = users.max()+1
    n_items = items.max()+1

    mu = values.mean()
    user_average = np.zeros(n_users)
    item_average = np.zeros(n_items)

    for iter in xrange(iters):
        if not reverse:
            user_average = np.bincount(users, weights=(values - mu - item_average[items])) / (np.bincount(users) + 1e-6)
            item_average = np.bincount(items, weights=(values - mu - user_average[users])) / (np.bincount(items) + 1e-6)
        else:
            item_average = np.bincount(items, weights=(values - mu - user_average[users])) / (np.bincount(items) + 1e-6)
            user_average = np.bincount(users, weights=(values - mu - item_average[items])) / (np.bincount(users) + 1e-6)

    return mu, user_average, item_average


def load_dataset(filename):
    data = np.loadtxt(filename)
    assert data.shape[1] >= 3
    return (data[:,0].astype('int32'),
            data[:,1].astype('int32'),
            data[:,2].astype('float32'))

if __name__ == '__main__':
    users, items, preferences = load_dataset(sys.argv[1])
    mu, user_average, item_average = shift((users, items), preferences, iters=10)
    svd_als((users, items), preferences=(preferences - mu - user_average[users] - item_average[items]),
            c_default=1e-2, p_default=10.0,
            rank=10, penalty=0.1, verbose=True, iters=30)
