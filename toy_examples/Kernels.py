import autograd.numpy as np
import time

def d_gaussian(x, y, sig):
    d_abs = -1 if x < y else 1
    return - np.abs(x - y) / sig * d_abs / np.sqrt(2 * np.pi * sig) * np.exp(- np.abs(x - y) ** 2 / (2 * sig))


def multi_gaussian(cov):
    def multi_gaussian_helper(inp, mu):
        """same mu for every datapoint given in ipn"""
        k = inp.shape[0]
        diff = (inp.T - mu).T
        cov_inv_prod = np.dot(np.linalg.inv(cov), diff)
        return 1 / np.sqrt(pow(2 * np.pi, k) * np.linalg.det(cov)) * np.exp(
            -0.5 * np.sum(diff*(cov_inv_prod), axis=0))
    return multi_gaussian_helper

def multi_gaussian_unnormalized(cov):
    def multi_gaussian_helper(inp, mu):
        """inp is [xs_1, xs_2, ..., xs_n] same mu for every datapoint given in ipn"""
        k = inp.shape[0]
        diff = (inp.T - mu).T
        return np.exp(-0.5 * np.sum(diff*(np.linalg.inv(cov).dot(diff)), axis=0))
    return multi_gaussian_helper


def grad_multi_gaussian(cov):
    def grad_gaussian_helper(inp, mu):
        """respect to x"""
        k = inp.shape[0]
        diff = (inp.T - mu).T

        mg = multi_gaussian(cov)(inp, mu)
        grad_term = - np.dot(np.linalg.inv(cov), diff)
        return mg * grad_term

    return grad_gaussian_helper


def hessian_multi_gaussian(cov):
    def hessian_gaussian_helper(inp, mu):
        """respect to x"""
        k = inp.shape[0]
        diff = (inp.T - mu).T

        mg = multi_gaussian(cov)(inp, mu)
        cov_inv_prod = np.dot(np.linalg.inv(cov), diff)
        grad_term = -np.linalg.inv(cov) + np.matmul(cov_inv_prod.T[:, :, np.newaxis], cov_inv_prod.T[:, np.newaxis, :])
        return (mg[:, np.newaxis, np.newaxis] * grad_term).T

    return hessian_gaussian_helper


def grad_trace_squared_hessian_multi_gaussian(cov):
    def helper(inp, mu):
        k = inp.shape[0]
        diff = (inp.T - mu).T

        mg = multi_gaussian(cov)(inp, mu)

        cov_inv_prod = np.dot(np.linalg.inv(cov), diff)

        C = -np.diag(np.linalg.inv(cov))[:, np.newaxis] + cov_inv_prod * cov_inv_prod
        A = 2 * np.sum(
            C.T[:, np.newaxis, :] * (np.linalg.inv(cov))[np.newaxis, :, :] * cov_inv_prod.T[:, np.newaxis, :],
            axis=-1).T
        B = - 2 * np.sum(np.matmul((C * C).T[:, :, np.newaxis], cov_inv_prod.T[:, np.newaxis, :]).T, axis=0)
        grad_term = 2 * A + B
        return mg ** 2 * grad_term

    return helper

def trace_multi_gaussian(cov):
    def helper(inp, mu):
        hessian = hessian_multi_gaussian(cov)(inp, mu)
        trace_squared = np.zeros(hessian.shape[2])
        for i in range(hessian.shape[0]):
            trace_squared += hessian[i][i] * hessian[i][i]
        return trace_squared
    return helper