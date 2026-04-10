"""Cost functions for power system operations.

This module provides various cost calculation functions for
generation, storage cycling, switching, and energy transactions.
"""

from typing import Sequence


def quadratic_cost(P: float, a: float, b: float, c: float) -> float:
    """Quadratic fuel/cycle cost: a*P^2 + b*P + c.

    Args:
        P: Power output (MW)
        a, b, c: Quadratic coefficients

    Returns:
        Cost value
    """
    return a * (P ** 2) + b * P + c


def polynomial_cost(P: float, coeffs: Sequence[float]) -> float:
    """General polynomial cost: sum_i coeffs[i] * P^i (ascending order).

    Args:
        P: Power output (MW)
        coeffs: Polynomial coefficients [c0, c1, c2, ...]

    Returns:
        Cost value
    """
    total = 0.0
    p_pow = 1.0
    for k, c in enumerate(coeffs):
        total += c * p_pow
        p_pow *= P
    return total


def piecewise_linear_cost(P: float, coefs: Sequence[float]) -> float:
    """Compute piecewise linear cost from (p0,f0,p1,f1,...) knots.

    Args:
        P: Power output (MW)
        coefs: Knot vector [p0, f0, p1, f1, ...]

    Returns:
        Interpolated cost value
    """
    assert len(coefs) % 2 == 0 and len(coefs) >= 4
    pts = sorted([(coefs[i], coefs[i + 1]) for i in range(0, len(coefs), 2)], key=lambda x: x[0])

    # clamp to end segments
    if P <= pts[0][0]:
        p0, f0 = pts[0]
        p1, f1 = pts[1]
    elif P >= pts[-1][0]:
        p0, f0 = pts[-2]
        p1, f1 = pts[-1]
    else:
        for (p0, f0), (p1, f1) in zip(pts[:-1], pts[1:]):
            if p0 <= P <= p1:
                break
    denom = (p1 - p0) if (p1 - p0) != 0 else 1e-9
    return f0 + (P - p0) * (f1 - f0) / denom


def cost_from_curve(P: float, coefs: Sequence[float]) -> float:
    """Interpret coefs as either quadratic or piecewise cost.

    Args:
        P: Power output (MW)
        coefs: If len==3, treat as quadratic [a,b,c]; else piecewise

    Returns:
        Cost value
    """
    if len(coefs) == 3:
        a, b, c = coefs
        return quadratic_cost(P, a, b, c)
    return piecewise_linear_cost(P, coefs)


def ramping_cost(P_prev: float, P_curr: float, up_cost: float = 0.0, down_cost: float = 0.0) -> float:
    """Charge for changing output.

    Args:
        P_prev: Previous power output
        P_curr: Current power output
        up_cost: Cost per MW of increase
        down_cost: Cost per MW of decrease

    Returns:
        Ramping cost
    """
    delta = P_curr - P_prev
    inc = max(0.0, delta) * up_cost
    dec = max(0.0, -delta) * down_cost
    return inc + dec


def switching_cost(changed: bool, cost_per_change: float) -> float:
    """Binary switching/toggling cost.

    Args:
        changed: Whether a switch occurred
        cost_per_change: Cost per switch event

    Returns:
        Switching cost
    """
    return float(cost_per_change) if changed else 0.0


def tap_change_cost(delta_steps: int, cost_per_step: float) -> float:
    """Cost proportional to OLTC steps moved.

    Args:
        delta_steps: Number of tap steps changed
        cost_per_step: Cost per step

    Returns:
        Tap change cost
    """
    return abs(int(delta_steps)) * float(cost_per_step)


def energy_cost(P_mw: float, price_per_mwh: float, discount: float) -> float:
    """Generic energy cost for net import/export.

    Args:
        P_mw: Net power (positive = import, negative = export)
        price_per_mwh: Energy price
        discount: Discount factor for export

    Returns:
        Energy cost (positive = payment, negative = credit)
    """
    if P_mw > 0:
        return P_mw * price_per_mwh
    else:
        return P_mw * price_per_mwh * discount
