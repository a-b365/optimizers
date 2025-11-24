from sgd import grad_w, grad_b, X, Y, error

def sgd_with_momentum(max_epochs):
    w, b, eta = -2, -2, 1.0 # eta is the learning rate
    prev_vw, prev_vb, beta = 0, 0, 0.9 # beta lies between 0 and 1
    for epoch in range(max_epochs):
        loss = error(w, b)
        print(f"Epoch {epoch} -> Loss {loss}")
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(x, w, b, y)
            db += grad_b(x, w, b, y)
        vw = beta * prev_vw + eta * dw
        vb = beta * prev_vb + eta * db
        w = w - vw
        b = b - vb

        prev_vw = vw
        prev_vb = vb


if __name__=="__main__":
    sgd_with_momentum(1000)
