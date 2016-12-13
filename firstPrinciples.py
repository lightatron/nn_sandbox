import numpy as np

def create_data(rows, cols):
    X = np.random.randn(rows, cols)*3 #+ 200
    v = np.random.randint(1, 100, (cols,))
    y = X.dot(v) # correct answer

    #noise = np.random.randn(rows, cols)
    #X_noise = X + noise

    return X, y, v

def error_func(y_true, y_train):
    resids = y_true - y_train
    err = np.square(resids).sum() * 0.5
    return err

def linear_activation_derivative(X, target, pred):
    # error function is the sum of suared errors * 0.5
    resids = target - pred
    #dEdw = -1.0 * resids.T.dot(X).T
    dEdw = -1 * X.T.dot(resids)

    #print dEdw
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

    for i in range(X.shape[0]):
        _y = X[i, :].dot(w)
        w = w  + learning_rate * (y[i] - _y) * X[i, :]
        print w

    return w


def find_v2(X, y, num_epochs=100, learning_rate=0.01):
    # simple solution to a 1 output neuron linear activation net
    rows, cols = X.shape
    w = np.random.randn(cols) # initialized weights
    preds = lambda _x, _w : np.dot(_x, _w)
    errs = lambda _y, __y : _y - __y
    deriv = lambda x, errs : x.T.dot(errs) * -1.0
    for e in range(num_epochs):
        _y = preds(X, w) #prediction of y
        error = errs(y, _y)
        grad = deriv(X, error)
        #import pdb; pdb.set_trace()
        w = w - learning_rate * grad
        print grad

    return w

def gradient_descent(x, y, iters, alpha):
    costs = []
    m = y.size # number of data points
    theta = np.random.rand(3) # random start
    history = [theta] # to store all thetas
    preds = []
    for i in range(iters):
        pred = np.dot(x, theta)
        error = pred - y
        cost = np.sum(error ** 2) / (2 * m)
        costs.append(cost)

        if i % 25 == 0: preds.append(pred)

        gradient = x.T.dot(error)/m
        print gradient
        #import pdb; pdb.set_trace()
        theta = theta - alpha * gradient  # update
        history.append(theta)

    print theta
    return history, costs, preds

def main():
    X, y, v = create_data(100, 3)
    print v
    v2 = find_v2(X, y, 10, 0.1)
    print v2
    #gradient_descent(X, y, 10, 0.1)

if __name__ == "__main__":
    main()
