import numpy as np

def create_data(rows, cols):
    X = np.random.randn(rows, cols)*3 #+ 200
    v = np.random.randint(1, 100, (cols,1))
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
    dEdw = -1.0 * resids.T.dot(X).T

    #print dEdw
    return dEdw

def find_v(X, y, num_epochs=100, learning_rate=0.01):
    # simple solution to a 1 output neuron linear activation net
    rows, cols = X.shape
    w = np.random.randn(cols,1) # initialized weights

    for e in range(num_epochs):
        _y = X.dot(w)
        w = w - learning_rate * linear_activation_derivative(X, y, _y)
        #print w

    return w

def find_v_online(X, y, num_epochs=100, learning_rate=0.01):
    # simple solution to a 1 output neuron linear activation net
    rows, cols = X.shape
    w = np.random.randn(cols,) # initialized weights

    for i in range(X.shape[0]):
        _y = X[i, :].dot(w)
        import pdb; pdb.set_trace()
        w = w + learning_rate * (y[i] - _y) * X[i, :]
        print w

    return w

def main():
    X, y, v = create_data(10, 3)
    print v
    v2 = find_v_online(X, y, 10, 0.1)

    print v2

if __name__ == "__main__":
    main()
