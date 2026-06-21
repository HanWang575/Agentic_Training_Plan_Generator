"""Deterministic calculators used by plans, APIs, and MCP tools."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any
from uuid import uuid4

from app.core.config import Settings
from app.program.schemas import CalculationResult


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculate_expression(expression: str) -> tuple[float, str]:
    """Evaluate arithmetic without names, calls, indexing, or attributes."""

    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return float(_OPERATORS[type(node.op)](evaluate(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("Expression exponent is too large.")
            return float(_OPERATORS[type(node.op)](left, right))
        raise ValueError("Expression contains unsupported syntax.")

    return evaluate(tree), expression


def estimate_repetition_max(weight: float, reps: int, method: str = "epley") -> tuple[float, str]:
    if weight <= 0 or reps < 1:
        raise ValueError("weight must be positive and reps must be at least 1.")
    if reps == 1:
        return float(weight), "weight"
    if method == "brzycki":
        if reps >= 37:
            raise ValueError("Brzycki estimation requires reps below 37.")
        return weight * 36 / (37 - reps), "weight * 36 / (37 - reps)"
    if method != "epley":
        raise ValueError("method must be 'epley' or 'brzycki'.")
    return weight * (1 + reps / 30), "weight * (1 + reps / 30)"


def calculate_percentage_load(one_rm: float, percentage: float) -> tuple[float, str]:
    if one_rm <= 0 or not 0 < percentage <= 100:
        raise ValueError("one_rm must be positive and percentage must be in (0, 100].")
    return one_rm * percentage / 100, "one_rm * percentage / 100"


def round_training_load(load: float, increment: float) -> tuple[float, str]:
    if load < 0 or increment <= 0:
        raise ValueError("load cannot be negative and increment must be positive.")
    return round(load / increment) * increment, "round(load / increment) * increment"


def calculate_volume_load(sets: int, reps: int, load: float) -> tuple[float, str]:
    if sets < 0 or reps < 0 or load < 0:
        raise ValueError("sets, reps, and load cannot be negative.")
    return float(sets * reps * load), "sets * reps * load"


def calculate_session_duration(
    work_seconds: float,
    rest_seconds: float,
    transition_seconds: float = 0,
) -> tuple[float, str]:
    if min(work_seconds, rest_seconds, transition_seconds) < 0:
        raise ValueError("duration inputs cannot be negative.")
    return work_seconds + rest_seconds + transition_seconds, "work_seconds + rest_seconds + transition_seconds"


def calculate_rest_interval(goal: str, compound: bool = True) -> tuple[float, str]:
    normalized = goal.strip().lower()
    if "strength" in normalized or "power" in normalized:
        value = 180 if compound else 120
    elif "endurance" in normalized or "circuit" in normalized:
        value = 30
    else:
        value = 120 if compound else 75
    return float(value), "goal-based deterministic rest table"


def calculate_distance_pace_time(
    distance_meters: float | None = None,
    pace_seconds_per_km: float | None = None,
    duration_seconds: float | None = None,
) -> tuple[float, str, str]:
    supplied = sum(item is not None for item in (distance_meters, pace_seconds_per_km, duration_seconds))
    if supplied != 2:
        raise ValueError("Provide exactly two of distance_meters, pace_seconds_per_km, and duration_seconds.")
    if distance_meters is None:
        value = duration_seconds / pace_seconds_per_km * 1000
        return value, "duration_seconds / pace_seconds_per_km * 1000", "m"
    if pace_seconds_per_km is None:
        value = duration_seconds / (distance_meters / 1000)
        return value, "duration_seconds / (distance_meters / 1000)", "s/km"
    value = pace_seconds_per_km * (distance_meters / 1000)
    return value, "pace_seconds_per_km * (distance_meters / 1000)", "s"


def calculate_weekly_frequency(sessions: int, weeks: int = 1) -> tuple[float, str]:
    if sessions < 0 or weeks <= 0:
        raise ValueError("sessions cannot be negative and weeks must be positive.")
    return sessions / weeks, "sessions / weeks"


def calculate_progression_increment(current_value: float, percentage: float) -> tuple[float, str]:
    if current_value < 0 or percentage < -100:
        raise ValueError("current_value cannot be negative and percentage cannot be below -100.")
    return current_value * (1 + percentage / 100), "current_value * (1 + percentage / 100)"


class CalculatorService:
    """Dispatch named calculators and apply configured rounding."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def run(
        self,
        tool_name: str,
        inputs: dict[str, Any],
        source_references: list[str] | None = None,
    ) -> CalculationResult:
        value, unit, formula, rounding = self._calculate(tool_name, inputs)
        if not math.isfinite(value):
            raise ValueError("Calculation result must be finite.")
        return CalculationResult(
            tool_call_id=f"tool_{uuid4().hex[:12]}",
            tool_name=tool_name,
            value=value,
            unit=unit,
            inputs=inputs,
            source_references=source_references or [],
            formula=formula,
            rounding_policy=rounding,
        )

    def _calculate(self, tool_name: str, inputs: dict[str, Any]) -> tuple[float, str, str, str]:
        if tool_name == "calculate_expression":
            value, formula = calculate_expression(str(inputs["expression"]))
            return value, str(inputs.get("unit", "count")), formula, "none"
        if tool_name == "estimate_repetition_max":
            value, formula = estimate_repetition_max(
                float(inputs["weight"]), int(inputs["reps"]), str(inputs.get("method", "epley"))
            )
            return round(value, 2), str(inputs.get("unit", "kg")), formula, "2 decimals"
        if tool_name == "calculate_percentage_load":
            value, formula = calculate_percentage_load(float(inputs["one_rm"]), float(inputs["percentage"]))
            return round(value, 2), str(inputs.get("unit", "kg")), formula, "2 decimals"
        if tool_name == "round_training_load":
            increment = float(inputs.get("increment", self.settings.program_barbell_load_rounding_kg))
            value, formula = round_training_load(float(inputs["load"]), increment)
            return value, str(inputs.get("unit", "kg")), formula, f"nearest {increment:g}"
        if tool_name == "calculate_volume_load":
            value, formula = calculate_volume_load(int(inputs["sets"]), int(inputs["reps"]), float(inputs["load"]))
            return value, str(inputs.get("unit", "kg")), formula, "none"
        if tool_name == "calculate_session_duration":
            value, formula = calculate_session_duration(
                float(inputs["work_seconds"]),
                float(inputs["rest_seconds"]),
                float(inputs.get("transition_seconds", 0)),
            )
            increment = self.settings.program_duration_rounding_seconds
            return round(value / increment) * increment, "s", formula, f"nearest {increment} seconds"
        if tool_name == "calculate_rest_interval":
            value, formula = calculate_rest_interval(str(inputs["goal"]), bool(inputs.get("compound", True)))
            increment = self.settings.program_rest_rounding_seconds
            return round(value / increment) * increment, "s", formula, f"nearest {increment} seconds"
        if tool_name == "calculate_distance_pace_time":
            value, formula, unit = calculate_distance_pace_time(
                inputs.get("distance_meters"), inputs.get("pace_seconds_per_km"), inputs.get("duration_seconds")
            )
            return round(value, 2), unit, formula, "2 decimals"
        if tool_name == "calculate_weekly_frequency":
            value, formula = calculate_weekly_frequency(int(inputs["sessions"]), int(inputs.get("weeks", 1)))
            return value, "sessions/week", formula, "none"
        if tool_name == "calculate_progression_increment":
            value, formula = calculate_progression_increment(
                float(inputs["current_value"]), float(inputs["percentage"])
            )
            return round(value, 2), str(inputs.get("unit", "count")), formula, "2 decimals"
        raise ValueError(f"Unknown calculator tool: {tool_name}")
