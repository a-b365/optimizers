import numpy as np
from sgd import grad_w, grad_b, error, X, Y

def do_rmsprop(max_epochs):

    # Initialization
    w, b, eta = -2, -2, 1.0
    v_w, v_b, eps, beta = 0, 0, 1e-6, 0.9

    for i in range(max_epochs):

        # Zero gradients
        dw , db = 0, 0

        for x, y in zip(X, Y):            
            # compute gradients
            dw += grad_w(x, w, b, y)
            db += grad_b(x, w, b, y)

        # compute intermediate values
        v_w = beta * v_w + (1 - beta) * (dw ** 2)
        v_b = beta * v_b + (1 - beta) * (db ** 2)

        # update parameters
        w = w - (eta * dw) / np.sqrt(v_w + eps)
        b = b - (eta * db) / np.sqrt(v_b + eps)

        loss = error(w, b)
        print(f"Epoch {i+1} -> Loss {loss}")
