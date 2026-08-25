from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict

class OptimizationStatus(str, Enum):
    """
    Outcome of a logical gate optimization.
    """
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    BOUND = "bound"

@dataclass
class OptimizationMetadata:
    """
    Statistics of the circuit construction and optimization.
    
    ``solutions`` contains ``[time, cost]`` pairs for each improving solution ordered by
    time found. Time is reported in seconds. The final entry corresponds to the returned
    circuit, if available.
    """

    status: OptimizationStatus
    time: float
    num_variables: int
    num_constraints: int
    solutions: List[List[float]]
    final_bound: float

    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "time": self.time,
            "num_variables": self.num_variables,
            "num_constraints": self.num_constraints,
            "solutions": self.solutions,
            "final_bound": self.final_bound,
        }

    @staticmethod
    def from_dict(d: Dict) -> OptimizationMetadata:
        return OptimizationMetadata(
            status=d.get("status", None),
            time=d.get("time", None),
            num_variables=d.get("num_variables", None),
            num_constraints=d.get("num_constraints", None),
            solutions=d.get("solutions", None),
            final_bound=d.get("final_bound", None),
        )