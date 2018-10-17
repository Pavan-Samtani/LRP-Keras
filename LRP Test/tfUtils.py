import numpy as np

#random_minibatches: Array(m, ...), Array(m, ...), int, int -> list(tuple(Array(m, ...), Array(m, ...)))
#Returns a list of shuffled minibatches of the specified size, and using the specified seed.
def random_minibatches(X_, Y_, minibatch_size, seed = 69):
    assert(X_.shape[0] == Y_.shape[0])
    m = X_.shape[0]
    order = np.random.permutation(m)
    num_complete_minibatches = m // minibatch_size
    X_ = X_[order, :]
    Y_ = Y_[order, :]
    j = 0
    minibatches = []
    for i in range(num_complete_minibatches):
        minibatches.append((X_[j : j+minibatch_size, :], Y_[j : j+minibatch_size, :]))
        j += minibatch_size
    if num_complete_minibatches != m/minibatch_size:
        minibatches.append((X_[j : m, :], Y_[j : m, :]))
    return minibatches
    
    