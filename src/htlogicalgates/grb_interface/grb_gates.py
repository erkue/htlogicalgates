import numpy as np
from numpy.typing import NDArray

from .grb_enviroment import *
from .grb_math_interface import *


def create_Perm(n: int, env: Enviroment) -> ExprMatrix:
    perm = ExprMatrix.create_zero(2*n, 2*n)
    one = Expression.create_const(1)
    for i in range(n):
        expr = Expression.create_const(0)
        for j in range(n):
            e = env.create_bin_expression()
            perm[i, j] = e
            perm[i+n, j+n] = e
            expr = expr + e
        env.add_equality_constraint(one, expr, False)
    for i in range(n):
        expr = Expression.create_const(0)
        for j in range(n):
            expr = expr + perm[j, i]
        env.add_equality_constraint(one, expr, False)
    return perm


def create_CZL(con: NDArray, env: Enviroment) -> ExprMatrix:
    n = np.shape(con)[0]
    czl = ExprMatrix.create_identity(2*n)
    assert (len(np.shape(con)) == 2 and n == np.shape(con)[1])
    for i in range(n):
        for j in range(i+1, n):
            if con[i, j] == 1:
                e = env.create_bin_expression()
                czl[i+n, j] = e
                czl[j+n, i] = e
    return czl


def create_SCL(n: int, env: Enviroment) -> ExprMatrix:
    SCL = ExprMatrix.create_zero(2*n, 2*n)
    for i in range(n):
        SCL[i, i] = env.create_bin_expression()
        SCL[i+n, i] = env.create_bin_expression()
        SCL[i, i+n] = env.create_bin_expression()
        SCL[i+n, i+n] = env.create_bin_expression()
        env.add_equality_constraint(Expression.create_const(1),
                                    SCL[i, i]*SCL[i+n, i+n] + SCL[i+n, i]*SCL[i, i+n],
                                    False)
    return SCL


def create_cons_SCL(n: int, env: Enviroment) -> ExprMatrix:
    SCL = ExprMatrix.create_zero(2*n, 2*n)
    for i in range(n):
        SCL[i, i] = env.create_bin_expression()
        SCL[i+n, i] = env.create_bin_expression()
        SCL[i, i+n] = env.create_bin_expression()
        SCL[i+n, i+n] = env.create_bin_expression()
        env.add_equality_constraint(Expression.create_const(1),
                                    SCL[i, i]*SCL[i+n, i+n] + SCL[i+n, i]*SCL[i, i+n],
                                    False)
        env.add_equality_constraint(Expression.create_const(1),
                                    SCL[i+n, i] + SCL[i+n, i+n],
                                    False)
        env.add_equality_constraint(Expression.create_const(1),
                                    SCL[i, i+n] + SCL[i+n, i+n],
                                    False)
    return SCL


def create_reduced_freedom_matrix(n: int, k: int, env: Enviroment) -> ExprMatrix:
    f = ExprMatrix.create_zero(n+k, n+k)
    for i in range(2*k, n+k):
        for j in range(n+k):
            f[i, j] = env.create_bin_expression()
    return f
