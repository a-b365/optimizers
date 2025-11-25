from sgd import grad_w, grad_b, X, Y, error

def batch_sgd(max_epochs):

    w, b, eta = -2, -2, 1.0
    for epoch in range(max_epochs):

        loss = error(w, b)

        print(f"Epoch {epoch} -> Loss {loss}")
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(x, w, b, y)
            db += grad_b(x, w, b, y)

            w -= eta*dw
            b -= eta*dw

if __name__ == "__main__":
    batch_sgd(1000)
