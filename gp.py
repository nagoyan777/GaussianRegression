#!/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from scipy.optimize import minimize, Bounds


class GP(object):
    def __init__(self):
        self.height = None
        self.sigma = None
        self.errfac = None
        self.x_train = None
        self.y_train = None
        self.K = None

    def train(self,
              x_train,
              y_train,
              height=4.0,
              sigma=1.0,
              errfac=0.2,
              train_params=True):
        self.x_train = x_train
        self.y_train = y_train

        params = self.height, self.sigma, self.errfac = [height, sigma, errfac]

        if train_params:

            def minusL(tau, x, y):
                theta = np.exp(tau)
                K = self.cov(x, x, theta[0], theta[1])

                K += self.err(x, theta[2])
                L = -np.log(
                    np.linalg.det(K)) - y.T @ np.matrix(K).I.tolist() @ y
                #                print(theta, -L)
                maxval = 1.0e6

                if L > maxval:
                    return maxval
                return float(-L)

            def gradL(tau, x, y):
                def _gradL(Kinv, Kinvy, dkdtau):
                    return -np.trace(Kinv @ dkdtau) + Kinvy.T @ dkdtau @ Kinvy

                theta = np.exp(tau)
                n = len(x)

                norm = self.get_norm(x, x)
                K = self.kernel(norm, height=theta[0], sigma=theta[1])

                dkdtau0 = K.copy()
                dkdtau1 = K * norm**2 / theta[1]
                dkdtau2 = theta[2] * np.eye(n)

                dkdtau = [dkdtau0, dkdtau1, dkdtau2]

                K = K + self.err(x, theta[2])
                Kinv = np.matrix(K).I.tolist()
                Kinvy = Kinv @ y
                grad_minusL = np.array(
                    [-_gradL(Kinv, Kinvy, _) for _ in dkdtau], dtype=float)
                msg = f'[{theta[0]:.4f} {theta[1]:.4f} {theta[2]:.4f}]'
                msg += f' {np.linalg.norm(grad_minusL):.4e}'
                msg += f' ({grad_minusL[0]:.2e} {grad_minusL[1]:.2e} {grad_minusL[2]:.2e})'
                print(msg)

                return grad_minusL


#            bounds = Bounds(np.log([1e-6, 1e-6, 1e-6]), np.log([10, 10, 10]))

            tau = np.log(params)
            res = minimize(
                minusL,
                tau,
                (x_train, y_train),
                jac=gradL,
                #                bounds=bounds,
                tol=1e-4,
                #                           method='cg',
                options={
                    'maxiter': 100,
                    'disp': True
                })
            print(np.exp(res.x))
            print(res.success)
            if not res.success:
                print(res)
                raise RuntimeError(
                    "Training does not converged. Try again with different initial values."
                )

            print(res)
            print(res.message)

            self.height, self.sigma, self.errfac = np.exp(res.x)

        self.K = self.cov(x_train, x_train, self.height, self.sigma)
        self.K += self.err(x_train, self.errfac)

    def pred(self, x):
        kk = self.cov(x, x, self.height, self.sigma) + self.err(x, self.errfac)
        k = self.cov(self.x_train, x, self.height, self.sigma)
        kK = k.T @ np.matrix(self.K).I.tolist()
        mu = kK @ self.y_train
        sigma = kk - kK @ k
        return mu, sigma

    def err(self, x, errfac):
        return errfac * np.eye(len(x))

    def cov(self, x0, x1, height=1.0, sigma=1.0):
        norm = self.get_norm(x0, x1)
        K = self.kernel(norm, height=height, sigma=sigma)
        return K

    def kernel(self, x, height=1.0, sigma=1.0):
        return height * np.exp(-x**2 / sigma)

    def get_norm(self, x0, x1):
        n0 = len(x0)
        n1 = len(x1)
        norm = np.zeros((n0, n1), dtype=float)
        for i0, _x0 in enumerate(x0):
            for i1, _x1 in enumerate(x1):
                norm[i0, i1] = self._norm(_x0, _x1)
        return norm

    def _norm(self, x0, x1):
        return np.linalg.norm(x1 - x0)

    def test(self):
        def f(x, a0=1.0, a1=1.0, a2=1.0):
            if len(x) == 2:
                x0 = x[0]
                x1 = x[1]
            elif x.shape[1] == 2:
                x0 = x[:, 0]
                x1 = x[:, 1]

            return a0 * np.cos(x0) + a1 * np.sin(x1) + a2 * np.cos(2 * x0)

        def plot_surface(f, xlim, nmesh=100, ax=None):
            mesh = np.linspace(*xlim, nmesh)
            x0, x1 = np.meshgrid(mesh, mesh)
            y = f([x0, x1])
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig = ax.figure
            ax.plot_surface(x0, x1, y, alpha=0.3)
            return fig, ax

        def get_training_data(f, ntrain=300, errfac=0.8, ax=None):
            twopi = 2 * np.pi
            x0 = twopi * np.random.rand(ntrain)
            x1 = twopi * np.random.rand(ntrain)
            x_train = np.array([x0, x1]).T
            y_train = f(x_train) + (np.random.rand(ntrain) - 0.5) * errfac
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig = ax.figure
            ax.scatter3D(x_train[:, 0], x_train[:, 1], y_train)
            return x_train, y_train

        nmesh = 100
        xlim = [0, 2 * np.pi]
        fig, ax = plot_surface(f, xlim, nmesh=nmesh)

        ntrain = 100
        x_train, y_train = get_training_data(f,
                                             ntrain=ntrain,
                                             errfac=2.0,
                                             ax=ax)

        self.train(x_train, y_train)

        print(self.height, self.sigma, self.errfac)

        npred = 100
        n = int(np.sqrt(npred))
        mesh = np.linspace(*xlim, n)
        x0, x1 = np.meshgrid(mesh, mesh)
        x_pred = np.array([np.ravel(x0), np.ravel(x1)]).T
        y_pred, sigma = self.pred(x_pred)
        print(x_pred.shape, y_pred.shape, sigma.shape)
        ax.scatter3D(x_pred[:, 0], x_pred[:, 1], y_pred)
        plt.show()

gp = GP()
gp.test()