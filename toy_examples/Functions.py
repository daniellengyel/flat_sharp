import autograd.numpy as np
from pypoly import Polynomial
from Kernels import *

from autograd import grad

# input format: xs[j][i] gives the value of the jth dimension of the ith point.
# e.g. the ith point in xs is given by xs[:, i]

def AckleyProblem(xs):
    out_shape = xs[0].shape
    a = np.exp(-0.2 * np.sqrt(1. / len(xs) * np.square(np.linalg.norm(xs, axis=0))))
    b = - np.exp(1. / len(xs) * np.sum(np.cos(2 * np.pi * xs), axis=0))
    return np.array(-20 * a + b + 20 + np.exp(1)).reshape(out_shape)


def GradAckleyProblem(xs):
    """del H/del xi = -20 * -0.2 * (xi * 1/n) / sqrt(1/n sum_j xj^2) * a + 2 pi sin(2 pi xi)/n * b"""
    out_shape = xs.shape
    a = np.exp(-0.2 * np.sqrt(1. / len(xs) * np.square(np.linalg.norm(xs, axis=0))))
    b = -np.exp(1. / len(xs) * np.sum(np.cos(2 * np.pi * xs), axis=0))
    a_p = -0.2 * (xs * 1. / len(xs)) / np.sqrt(1. / len(xs) * np.square(np.linalg.norm(xs, axis=0)))
    b_p = -2 * np.pi * np.sin(2 * np.pi * xs) / len(xs)
    return np.nan_to_num(
        -20 * a_p * a + b_p * b).reshape(out_shape)  # only when norm(x) == 0 do we have nan and we know the grad is zero there


def QuadraticFunctionInit(A, b):
    def QuadraticFunction(xs):
        out_shape = xs[0].shape
        xs = np.array([x.flatten() for x in xs])
        return (np.diag(np.dot(xs.T, np.dot(A, xs))) + b).reshape(out_shape)
    return QuadraticFunction

def GradQuadraticFunctionInit(A):
    def GradQuadraticFunction(xs):
        out_shape = xs.shape
        xs = np.array([x.flatten() for x in xs])
        grad = np.dot(xs.T, A + A.T).T
        return np.array([g for g in grad]).reshape(out_shape)
    return GradQuadraticFunction


def Gibbs(x, U, sig):
    return np.exp(-U(np.array(x)) / sig ** 2)


def GradGibbs(x, U, grad_U, sig):
    return -grad_U(x) * 1./sig**2 * Gibbs(x, U, sig)


def grad_one_D_shallow(inp):
    out_shape = inp.shape
    inp = inp.flatten()

    X = [[0, 0],
         [0.25, 0.05], [-0.25, 0.05],
         [0.5, 0.15], [-0.5, 0.15],
         [1, 1], [-1, 1],
         [1.4, 0.2], [-1.4, 0.2],
         [1.5, 0], [-1.5, 0],
         [2, 1], [-2, 1],
         [2.2, 1], [-2.2, 1],
         [2.5, 0.5], [-2.5, -0.5],
         ]

    equations = np.array([[point[0] ** i for i in range(len(X))] for point in X])
    values = np.array([point[1] for point in X])
    coefficients = np.linalg.solve(equations, values)

    coefficients = coefficients[1:]
    coefficients = [coefficients[i] * (i + 1) for i in range(len(coefficients))]

    p = Polynomial(*coefficients)

    res = [p(x) for x in inp]

    return np.array(res).reshape(out_shape)


def one_D_shallow(inp):
    """
    Will apply poly function to every entry of input array

    :param inp: numpy
    :return: numpy
    """
    out_shape = inp.shape[1]
    inp = inp.flatten()

    X = [[0, 0],
         [0.25, 0.05], [-0.25, 0.05],
         [0.5, 0.15], [-0.5, 0.15],
         [1, 1], [-1, 1],
         [1.4, 0.2], [-1.4, 0.2],
         [1.5, 0], [-1.5, 0],
         [2, 1], [-2, 1],
         [2.2, 1], [-2.2, 1],
         [2.5, 0.5], [-2.5, -0.5],

         ]

    equations = np.array([[point[0] ** i for i in range(len(X))] for point in X])
    values = np.array([point[1] for point in X])
    coefficients = np.linalg.solve(equations, values)

    p = Polynomial(*coefficients)

    res = [p(x) for x in inp]

    return np.array(res).reshape(out_shape)

def quad_sin(inp, params=None):
    out_shape = inp.shape
    inp = inp.flatten()

    epsilon = params['epsilon']
    res = (inp*np.sin(inp/epsilon)+0.1*inp)**2
    res = res.reshape(out_shape)
    return np.sum(res, axis=0)


def grad_quad_sin(inp, params=None):
    out_shape = inp.shape
    inp = inp.flatten()

    epsilon = params['epsilon']
    res = 2 * (inp*np.sin(inp/epsilon)+0.1*inp) * (np.sin(inp/epsilon) + inp * 1./epsilon * np.cos(inp/epsilon) + 0.1)
    return res.reshape(out_shape)


def flat_sharp_gaussian(inp):
    dim = inp.shape[0]

    flat_gaussian = multi_gaussian(10 * np.eye(dim))
    sharp_gaussian = multi_gaussian(0.7 * np.eye(dim))
    very_flat_gaussian = multi_gaussian(30 * np.eye(dim))  # to keep things rolling towards center
    return -(flat_gaussian(inp, 5 * np.ones(dim)) + sharp_gaussian(inp, -5 * np.ones(dim)) + 2*very_flat_gaussian(inp,
                                                                                                                np.zeros(
                                                                                                                    dim)))
def grad_flat_sharp_gaussian(inp):
    dim = inp.shape[0]

    flat_gaussian_grad = grad_multi_gaussian(10 * np.eye(dim))
    sharp_gaussian_grad = grad_multi_gaussian(0.7 * np.eye(dim))
    very_flat_gaussian_grad = grad_multi_gaussian(30 * np.eye(dim))  # to keep things rolling towards center
    return -(flat_gaussian_grad(inp, 5 * np.ones(dim)) + sharp_gaussian_grad(inp, -5 * np.ones(dim)) + 2*very_flat_gaussian_grad(inp,
                                                                                                                                 np.zeros(
                                                                                                                                     dim)))

def flat_sharp_hill_gaussian(inp):
    dim = inp.shape[0]

    flat_gaussian = multi_gaussian(10 * np.eye(dim))
    sharp_gaussian = multi_gaussian(0.7 * np.eye(dim))
    very_flat_gaussian = multi_gaussian(30 * np.eye(dim))  # to keep things rolling towards center
    hill_gaussian = multi_gaussian(3 * np.eye(dim))
    boundary_hill = multi_gaussian(0.5 * np.eye(dim))
    return 5*boundary_hill(inp, -15*np.ones(dim)) + 5*boundary_hill(inp, 25*np.ones(dim)) + hill_gaussian(inp, 17.5 * np.ones(dim)) - (flat_gaussian(inp, 5 * np.ones(dim)) + sharp_gaussian(inp,
                                                                                                             -5 * np.ones(
                                                                                                                 dim)) + 2 * very_flat_gaussian(inp, np.zeros(dim)))
def grad_flat_sharp_hill_gaussian(inp):
    dim = inp.shape[0]

    flat_gaussian_grad = grad_multi_gaussian(10 * np.eye(dim))
    sharp_gaussian_grad = grad_multi_gaussian(0.7 * np.eye(dim))
    very_flat_gaussian_grad = grad_multi_gaussian(30 * np.eye(dim))  # to keep things rolling towards center
    hill_gaussian_grad = grad_multi_gaussian(3 * np.eye(dim))
    boundary_hill_grad = grad_multi_gaussian(0.5 * np.eye(dim))
    return 5*boundary_hill_grad(inp, -15*np.ones(dim)) + 5*boundary_hill_grad(inp, 25*np.ones(dim)) + hill_gaussian_grad(inp, 17.5 * np.ones(dim)) - (flat_gaussian_grad(inp, 5 * np.ones(dim)) + sharp_gaussian_grad(inp, -5 * np.ones(dim)) + 2*very_flat_gaussian_grad(inp,
                                                                                                                                 np.zeros(
                                                                                                                                     dim)))

def gaussian_sum(params):
    """params: array of tuples x_i s.t. x_i[0] = mean, x_i[1] = cov, x_i[2] = scale"""
    assert len(params) > 0
    def helper(inp):
        dim = inp.shape[0]
        res = 0
        for i in range(len(params)):
            temp_param = params[i]
            res += temp_param[2] * multi_gaussian(temp_param[1] * np.eye(dim))(inp, temp_param[0] * np.ones(dim))
        return res
    return helper

def grad_gaussian_sum(params):
    """params: array of tuples x_i s.t. x_i[0] = mean, x_i[1] = cov, x_i[2] = scale"""
    assert len(params) > 0
    def helper(inp):
        dim = inp.shape[0]
        res = 0
        for i in range(len(params)):
            temp_param = params[i]
            res += temp_param[2] * grad_multi_gaussian(temp_param[1] * np.eye(dim))(inp, temp_param[0] * np.ones(dim))
        return res
    return helper

def trace_hessian_gaussian_sum(params):
    """params: array of tuples x_i s.t. x_i[0] = mean, x_i[1] = cov, x_i[2] = scale"""
    assert len(params) > 0
    def helper(inp):
        dim = inp.shape[0]
        res = 0
        for i in range(len(params)):
            temp_param = params[i]
            res += abs(temp_param[2]) * trace_multi_gaussian(temp_param[1] * np.eye(dim))(inp, temp_param[0] * np.ones(dim))
        return res
    return helper

def grad_trace_hessian_gaussian_sum(params):
    """params: array of tuples x_i s.t. x_i[0] = mean, x_i[1] = cov, x_i[2] = scale"""
    assert len(params) > 0
    def helper(inp):
        dim = inp.shape[0]
        res = 0
        for i in range(len(params)):
            temp_param = params[i]
            res += abs(temp_param[2]) * grad_trace_squared_hessian_multi_gaussian(temp_param[1] * np.eye(dim))(inp, temp_param[0] * np.ones(dim))
        return res
    return helper


def SGDL(x, L, gamma, alpha, SGDL_step_size, var, grad_f, f):
    x_p = x
    mu = x

    for l in range(L):
        dx_p = (grad_f(x_p) - gamma*(x - x_p))
        x_p = x_p - 0.5 * SGDL_step_size * dx_p + np.sqrt(SGDL_step_size) * var * np.random.normal(size=x_p.shape)
        mu = (1 - alpha)* mu + alpha * x_p

    return gamma*(x - mu)


def approx_local_entroy_grad(inp, f, step_size, gamma):
    """We evaluate it with bounds given by +/- 3sig = +/- 3 sqrt(1/gamma)"""

    sig = np.sqrt(1 / float(gamma))
    num_sig = 4
    x_p = np.linspace(inp - num_sig * sig, inp + num_sig * sig, (2 * num_sig * sig / step_size)).T

    x_p_shape = x_p.shape
    f_outs = f(np.array([x_p.reshape(-1)])).reshape(x_p_shape)

    ws = np.exp(- f_outs - 0.5 * gamma * (x_p - inp.T[:, :, np.newaxis]) ** 2)
    approx_e = np.sum(x_p * ws, axis=2) / np.sum(ws, axis=2)

    return np.array([gamma * (inp.T - approx_e)])


def approx_local_entropy_F(inp, f, step_size, gamma):
    """We evaluate it with bounds given by +/- 3sig = +/- 3 sqrt(1/gamma)"""

    sig = np.sqrt(1 / float(gamma))
    num_sig = 4
    x_p = np.linspace(inp - num_sig * sig, inp + num_sig * sig, (2 * num_sig * sig / step_size)).T

    x_p_shape = x_p.shape
    f_outs = f(np.array([x_p.reshape(-1)])).reshape(x_p_shape)

    ws = np.exp(- f_outs - 0.5 * gamma * (x_p - inp.T[:, :, np.newaxis]) ** 2)
    approx_integral = np.sum(ws, axis=2)


    # xs = np.random.normal(loc=mu, scale=1/float(gamma), size=(1, N))
    # out = np.sum(np.exp(-f(xs)))/float(N) #* np.sqrt(2 * np.pi * var)
    return -np.log(approx_integral)



if __name__ == "__main__":
    x = np.array([[0, 1], [0, 2]])
    g_out = Gibbs(x, AckleyProblem, 1)
    grad_g_out = GradGibbs(x, AckleyProblem, GradAckleyProblem, 1)
    print(g_out)
  