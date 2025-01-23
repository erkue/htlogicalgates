from __future__ import annotations
from typing import List, Tuple, Iterable
from itertools import product
from copy import deepcopy

import gurobipy as gp
import numpy as np
from numpy.typing import NDArray


class Expression:
    @staticmethod
    def create_var(var: gp.Var) -> Expression:
        return Expression([1], [[var]], 0)

    @staticmethod
    def create_const(c: int) -> Expression:
        return Expression([], [], c)

    def __init__(self, p: List[int], n: List[List[gp.Var]], c: int) -> None:
        self._prefac = p
        self._nums = n
        self._const = c

    def __add__(self, other: Expression):
        assert (isinstance(other, Expression))
        return Expression(self._prefac + other._prefac,
                          self._nums + other._nums,
                          self._const + other._const)

    def __mul__(self, other: Expression):
        assert (isinstance(other, Expression))
        if self.is_zero() or other.is_zero():
            return Expression.create_const(0)
        prefac = [a*b for a, b in product(self._prefac, other._prefac)]
        nums = [a+b for a, b in product(self._nums, other._nums)]
        if self._const != 0:
            prefac += [self._const*a for a in other._prefac]
            nums += other._nums
        if other._const != 0:
            prefac += [other._const*a for a in self._prefac]
            nums += self._nums
        return Expression(prefac, nums, self._const*other._const)

    def __repr__(self) -> str:
        s = ""
        for fac, nums in zip(self._prefac, self._nums):
            s += str(fac) if fac != 1 else ""
            for n in nums:
                s += f"{{{n}}}"
            s += "+"
        if self._const != 0 or len(s) == 0:
            return s + f"{self._const}"
        else:
            return s[:-1]

    def is_const(self) -> bool:
        return len(self._prefac) == 0

    def is_zero(self) -> bool:
        return self.is_const() and self._const == 0

    def copy(self) -> Expression:
        expr: List[List[gp.Var]] = []
        for i in self._nums:
            expr.append([])
            for j in i:
                expr[-1].append(j)
        return Expression(deepcopy(self._prefac), expr, self._const)


class ExprMatrix:
    @staticmethod
    def create_zero(rows: int, columns: int) -> ExprMatrix:
        expr: List[List[Expression]] = []
        for i in range(rows):
            expr.append([])
            for j in range(columns):
                expr[-1].append(Expression.create_const(0))
        return ExprMatrix(expr)

    @staticmethod
    def create_identity(n: int) -> ExprMatrix:
        expr: List[List[Expression]] = []
        for i in range(n):
            expr.append([])
            for j in range(n):
                expr[-1].append(Expression.create_const(1 if i == j else 0))
        return ExprMatrix(expr)

    @staticmethod
    def create_from_array(arr: NDArray) -> ExprMatrix:
        expr: List[List[Expression]] = []
        for i in range(len(arr)):
            expr.append([])
            for j in range(len(arr[0])):
                expr[-1].append(Expression.create_const(arr[i, j]))
        return ExprMatrix(expr)

    def create_expanded_dims(self, rows: int, columns: int, fill_value: Expression = Expression.create_const(0)):
        e = ExprMatrix.create_zero(rows, columns)
        for i in range(rows):
            for j in range(columns):
                if i < self._ROWS and j < self._COLUMNS:
                    e[i, j] = self[i, j].copy()
                else:
                    e[i, j] = fill_value.copy()
        return e

    def transpose(self) -> ExprMatrix:
        e = ExprMatrix.create_zero(self._ROWS, self._COLUMNS)
        for i in range(self._ROWS):
            for j in range(self._COLUMNS):
                e[i, j] = self[j, i]
        return e

    def __init__(self, expr: List[List[Expression]]) -> None:
        self._ROWS = len(expr)
        self._COLUMNS = len(expr[0])
        self._expr = expr

    def __matmul__(self, other: ExprMatrix) -> ExprMatrix:
        assert (self._COLUMNS == other._ROWS)
        expr: List[List[Expression]] = []
        for i in range(self._ROWS):
            expr.append([])
            for j in range(other._COLUMNS):
                expr[-1].append(sum([self[i, k]*other[k, j]
                                for k in range(self._COLUMNS)], start=Expression.create_const(0)))
        return ExprMatrix(expr)

    def __add__(self, other: ExprMatrix) -> ExprMatrix:
        expr: List[List[Expression]] = []
        for i in range(self._ROWS):
            expr.append([])
            for j in range(other._COLUMNS):
                expr[-1].append(self[i, j] + other[i, j])
        return ExprMatrix(expr)

    def __getitem__(self, key: Tuple[int, int]):
        return self._expr[key[0]][key[1]]

    def __setitem__(self, key: Tuple[int, int], val: Expression):
        self._expr[key[0]][key[1]] = val

    def __repr__(self) -> str:
        return self._expr.__repr__()
