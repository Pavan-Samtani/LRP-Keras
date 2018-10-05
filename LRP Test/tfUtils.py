import numpy as np

#random_minibatches: Array(m, ...), Array(m, ...), int, int -> list(tuple(Array(m, ...), Array(m, ...)))
#Returns a list of shuffled minibatches of the specified size, and using the specified seed.
def random_minibatches(X, Y, minibatch_size, seed = 69):
    assert(X.shape[0] == Y.shape[0])
    m = X.shape[0]
    order = list(np.random.permutation(m))
    num_complete_minibatches = m // minibatch_size
    X = X[order]
    Y = Y[order]
    j = 0
    minibatches = []
    for i in range(num_complete_minibatches):
        minibatches.append((X[j : j+minibatch_size], Y[j : j+minibatch_size]))
        j += minibatch_size
    if num_complete_minibatches != m/minibatch_size:
        minibatches.append((X[j : m-1], Y[j : m-1]))
    return minibatches
    
    