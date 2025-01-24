from __future__ import annotations

from typing import Dict, Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy.typing import NDArray

from .grb_math_interface import *


class Enviroment:
    NAME_BIN = "bin"
    NAME_CONST = "cst"
    NAME_CONST_CONSTR = "constraint"

    NAME_AND = "and"

    #### Creating model ####

    def __init__(self, log_to_console: bool = False, log_file: str = "",
                 gurobi: Dict[Any, Any] = {}) -> None:
        self._bin_counter = 0
        self._predef_counter = 0

        self.enviroment = gp.Env(logfilename=log_file, empty=True)

        self.enviroment.setParam("LogToConsole", log_to_console)
        self.enviroment.start()
        self.model = gp.Model(env=self.enviroment)
        self.model.setParam(GRB.Param.Presolve, 2)
        self.model.setParam(GRB.Param.Cuts, 2)
        for key, val in gurobi.items():
            self.model.setParam(key, val)

    def set_time_limit(self, time_limit: float):
        self.model.setParam(GRB.Param.TimeLimit, time_limit)

    def create_bin_var(self) -> gp.Var:
        self._bin_counter += 1
        v = self.model.addVar(
            vtype=GRB.BINARY, name=Enviroment.NAME_BIN + str(self._bin_counter))
        return v

    def create_bin_expression(self) -> Expression:
        return Expression.create_var(self.create_bin_var())

    def create_predef_bin_expression(self) -> Tuple[Expression, int]:
        i = self.create_predef_var()
        return Expression.create_var(i), -self._predef_counter

    def create_predef_bin_matrix(self, rows: int, columns: int) -> Tuple[ExprMatrix, NDArray]:
        ids: List[List[int]] = []
        expr: List[List[Expression]] = []
        for i in range(rows):
            ids.append([])
            expr.append([])
            for j in range(columns):
                a, b = self.create_predef_bin_expression()
                expr[-1].append(a)
                ids[-1].append(b)
        return ExprMatrix(expr), np.array(ids, dtype=np.int32)

    def create_predef_var(self) -> gp.Var:
        self._predef_counter += 1
        v = self.model.addVar(
            vtype=GRB.BINARY, name=Enviroment.NAME_CONST + str(self._predef_counter))
        return v

    def set_predef_var(self, value: int, id: int):
        c = self.model.getConstrByName(Enviroment.NAME_CONST_CONSTR + str(id))
        if c is not None:
            self.model.remove(c)
        v = self._get_var_by_id(id)
        self.model.addConstr(
            int(value) == v, name=Enviroment.NAME_CONST_CONSTR + str(id))

    def set_many_predef_var(self, values: NDArray, ids: NDArray):
        self.model.update()
        for i in range(len(values)):
            for j in range(len(values[0])):
                self.set_predef_var(values[i, j], ids[i, j])

    def add_equality_constraint_mat(self, lhs: ExprMatrix, rhs: ExprMatrix, add_slack: bool = True):
        assert (lhs._COLUMNS == rhs._COLUMNS and lhs._ROWS == rhs._ROWS)
        for i in range(lhs._ROWS):
            for j in range(lhs._COLUMNS):
                self.add_equality_constraint(lhs[i, j], rhs[i, j], add_slack)

    def add_equality_constraint(self, lhs: Expression, rhs: Expression, add_slack: bool = True):
        if add_slack:
            slack = self._get_slack_var()
            self.model.addConstr(sum([a*self._add_AND_constraint(b) for a, b in zip(lhs._prefac, lhs._nums)], start=lhs._const) == 2*slack +
                                 sum([a*self._add_AND_constraint(b) for a, b in zip(rhs._prefac, rhs._nums)], start=rhs._const))
        else:
            self.model.addConstr(sum([a*self._add_AND_constraint(b) for a, b in zip(lhs._prefac, lhs._nums)], start=lhs._const) ==
                                 sum([a*self._add_AND_constraint(b) for a, b in zip(rhs._prefac, rhs._nums)], start=rhs._const))

    def set_target_function(self, expr: Expression):
        self.model.setObjective(sum([a*self._add_AND_constraint(b)
                                for a, b in zip(expr._prefac, expr._nums)], start=expr._const), GRB.MINIMIZE)

    def get_model(self) -> gp.Model:
        return self.model

    def solve(self):
        self.model.optimize()

    #### After solving ####

    def evaluate_expression(self, expr: Expression) -> int:
        return expr._const + sum([a*np.prod([round(i.X) for i in b]) for a, b in zip(expr._prefac, expr._nums)])

    def evaluate_matrix(self, mat: ExprMatrix) -> NDArray:
        return np.array([[self.evaluate_expression(j) for j in i] for i in mat._expr], dtype=np.int32)

    def has_solution(self) -> bool:
        return self.model.SolCount > 0

    def get_status(self) -> str:
        if self.has_solution():
            if self.model.Status == GRB.OPTIMAL:
                return "Optimal"
            else:
                return "Bound " + str(self.model.ObjBound)
        else:
            if self.model.Status == GRB.INFEASIBLE:
                return "Infeasible"
            else:
                return "Time out"

    def get_runtime(self) -> float:
        return self.model.Runtime

    def get_work(self) -> float:
        return self.model.Runtime

    #### Internal functions ####

    def _add_AND_constraint(self, vars: List[gp.Var]) -> gp.Var:
        if len(vars) == 1:
            return vars[0]
        if len(vars) == 0:
            raise ValueError("List of variables has size zero!")
        v = self.create_bin_var()
        self.model.addGenConstrAnd(
            v, vars, name=Enviroment.NAME_AND + str(self._bin_counter))
        return v

    def _get_var_by_id(self, id: int) -> gp.Var:
        try:
            if id > 0:
                v = self.model.getVarByName(Enviroment.NAME_BIN + str(id))
            elif id < 0:
                v = self.model.getVarByName(Enviroment.NAME_CONST + str(-id))
        except gp.GurobiError:
            self.model.update()
            return self._get_var_by_id(id)
        if v is None:
            self.model.update()
            return self._get_var_by_id(id)
        return v

    def _get_slack_var(self) -> gp.Var:
        return self.model.addVar(-float("inf"), float("inf"), vtype=GRB.INTEGER)
