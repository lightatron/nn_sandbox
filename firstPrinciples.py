import numpy as np

''' Just a bunch of stuff/ways to calculate a linear system with one
output neuron and no bias. Trick was my learning rate was far too high.
Thought would converge anyway. Didn't.
'''

def create_data(rows, cols):
    X = np.random.randn(rows, cols)*3# + 200
    v = np.random.randint(1, 100, (cols,))
    y = X.dot(v) # correct answer

    noise = np.random.randn(rows, cols) * 0.25
    X_noise = X + noise

    return X_noise, y, v


def error_func(y_true, y_train):
    resids = y_true - y_train
    err = np.square(resids).sum() * 0.5
    return err


def linear_activation_derivative(X, target, pred):
    # error function is the sum of suared errors * 0.5
    resids = target - pred
    dEdw = -1 * X.T.dot(resids) / X.shape[0]
    return dEdw


def find_v(X, y, num_epochs=100, learning_rate=0.01):
    # simple solution to a 1 output neuron linear activation net
    rows, cols = X.shape
    w = np.random.randn(cols,) # initialized weights
    for e in range(num_epochs):
        _y = X.dot(w) #prediction of y
        w = w - learning_rate * linear_activation_derivative(X, y, _y)
        #print w

    return w


def find_v_online(X, y, num_epochs=100, learning_rate=0.01):
    # simple solution to a 1 output neuron linear activation net
    rows, cols = X.shape
    w = np.random.randn(cols,) # initialized weights
    for i in range(num_epochs):
        for i in range(X.shape[0]):
            _y = X[i, :].dot(w)
            w = w  + learning_rate * (y[i] - _y) * X[i, :]

    return w


def find_v2(X, y, num_epochs=100, learning_rate=0.01):
    # simple solution to a 1 output neuron linear activation net
    rows, cols = X.shape
    w = np.random.randn(cols) # initialized weights
    preds = lambda _x, _w : np.dot(_x, _w)
    errs = lambda _y, __y : _y - __y
    deriv = lambda x, errs : x.T.dot(errs) * -1.0
    for e in range(num_epochs):
        p = preds(X, w) #prediction of y
        error = errs(y, p)
        grad = deriv(X, error) / rows
        w = w - learning_rate * grad

    return w


def main():
    X, y, v = create_data(5000, 5)
    print v
    v2 = find_v(X, y, 10000, 0.001)
    print v2
    v2 = find_v2(X, y, 10000, 0.001)
    print v2
    v2 = find_v_online(X, y, 1000, 0.001)
    print v2

if __name__ == "__main__":
    main()
