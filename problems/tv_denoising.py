import numpy as np
import cvxpy as cp
from skimage import data

DATA = [
    data.chelsea(),
    data.astronaut(),
    data.coffee(),
    data.immunohistochemistry(),
    data.logo(),
    data.brick(),
    data.camera(),
    data.grass(),
]


def tv_denoising_cvxpy(n):
    img = DATA[n]
    if img.ndim == 2:
        rows, cols = img.shape
        colors = 1
    else:
        rows, cols, colors = img.shape

    sigma = 20
    noise = sigma * np.random.randn(rows, cols, colors)
    corr = img + noise
    corr = np.clip(corr, 0, 255)
    lam = 0.1

    variables = []
    least_squares = 0

    for i in range(colors):
        U = cp.Variable(shape=(rows, cols))
        variables.append(U)
        least_squares += 0.5 * lam * cp.sum_squares(U - corr[:, :, i])

    prob = cp.Problem(cp.Minimize(cp.tv(*variables) + least_squares))

    return prob
