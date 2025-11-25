from sgd import grad_w, grad_b, X, Y, error

def do_line_search_gradient_descent(max_epochs):
    w, b, etas = -2, -2, [0.1, 0.5, 1, 2, 10]  # eta is the learning rate
    
    for epoch in range(max_epochs):
        loss = error(w, b)
        print(f"Epoch {epoch} -> Loss {loss}")

        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(x, w, b, y)
            db += grad_b(x, w, b, y)

        min_error = 10000
        for eta in etas:
            temp_w = w - eta * dw
            temp_b = b - eta * db

            if error(temp_w, temp_b) < min_error:
                best_w = temp_w
                best_b = temp_b
                min_error = error(best_w, best_b)


        w = best_w
        b = best_b



if __name__=="__main__":
    sgd_with_momentum(1000)
