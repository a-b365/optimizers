import numpy as np

X = [0.5, 2.5]
Y = [0.2, 0.9]

# Sigmoid Activation Function
def f(x, w, b):
    return 1/(1 + np.exp(-(w * x + b)))

def error(w, b):
    err = 0.0
    for x, y in zip(X, Y):
        y_pred = f(x, w, b)
        temp = (y_pred - y)**2
        err += temp
    return 0.5 * err

def grad_b(x, w, b, y):
    y_pred = f(w, x, b)
    return (y_pred-y)*y_pred*(1-y_pred)

def grad_w(x, w, b, y):
    y_pred = f(x, w, b)
    return (y_pred-y)*y_pred*(1-y_pred)*x

def do_gradient_descent():
    w, b, eta, max_epochs = -2, -2, 1.0, 1000

    for epoch in range(max_epochs):
        loss = error(w, b)
        print(f"Epoch {epoch} -> Loss {loss}")
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(x, w, b, y)
            db += grad_b(x, w, b, y)

        w = w - eta * dw
        b = b - eta * db

if __name__ == "__main__":
    do_gradient_descent()



