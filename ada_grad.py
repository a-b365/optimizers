import numpy as np
from sgd import grad_w, grad_b, X, Y, error


def ada_grad(max_epochs):

    # Initialization
    w, b, eta = -2, -2, 0.1
    v_w , v_b, eps = 0, 0, 1e-6

    for i in range(len(max_epochs)):

        # Zero gradients
        dw, db = 0, 0

        for x, y in zip(X, Y):
            loss = error(w, b)
            print(f"Epoch {i} -> Loss {loss}")
            # compute gradients
            dw += grad_w(x, w, b, y)
            dy += grad_b(x, w, b, y) 
           
        # compute intermediate values

        # update parameters
        w = w - (eta * dw)/(np.sqrt(v_w + eps)) 
        b = b - (eta * db)/(np.sqrt(v_w + eps))



if __name__ == "__main__":
    ada_grad(1000)