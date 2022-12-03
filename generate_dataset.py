import numpy as np
import csv

def logistic(z):
    return 1 / (1 + np.exp(-z))

def save(outfile, X, y):
    data = np.hstack([y.reshape(y.size, -1), X])
    with open(outfile, "w") as f:
        writer = csv.writer(f, delimiter=' ')
        for row in data:
            writer.writerow(row)

def generate(n, theta, deterministic=False, seed=1):
    # Number of features.
    dim = len(theta)

    # Generate feature values from U[0,1].
    np.random.seed(seed)
    X = np.random.rand(n, dim)

    # Calculate logits.
    z = np.dot(X, theta)
    # Calculate probabilities.
    prob = logistic(z)

    if deterministic:
        # Assign classes deterministically.
        y = np.where(prob.flatten() >= 0.5, 1, 0)
    else:
        # Generate labels by sampling from Bernoulli(prob)
        y = np.random.binomial(1, prob.flatten())
    
    return X, y
   
if __name__ == "__main__":
    # True theta coefficients.
    theta = np.random.rand(10)
    X, y = generate(5300, theta)
    save("default.csv", X, y)

