from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction
from typing import Iterable


@dataclass(frozen=True)
class Estimate:
    value: Fraction
    variance: Fraction
    n: int

    def as_dict(self) -> dict:
        with localcontext() as context:
            context.prec = 50
            value = Decimal(self.value.numerator) / Decimal(self.value.denominator)
            variance = Decimal(self.variance.numerator) / Decimal(self.variance.denominator)
            half_width = Decimal("1.96") * variance.sqrt()
            percent = (100 * value).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            ci_percent = (100 * half_width).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        return {
            "value": float(self.value),
            "ci95_half_width": float(half_width),
            "n": self.n,
            "percent": str(percent),
            "ci95_percent": str(ci_percent),
            "display": f"{percent} ± {ci_percent}",
        }


def binomial(verdicts: Iterable[bool]) -> Estimate:
    values = tuple(verdicts)
    if not values or any(type(value) is not bool for value in values):
        raise ValueError("a measured binomial sample requires nonempty boolean verdicts")
    n = len(values)
    p = Fraction(sum(values), n)
    return Estimate(p, p * (1 - p) / n, n)


def macro_average(estimates: Iterable[Estimate]) -> Estimate:
    values = tuple(estimates)
    if not values:
        raise ValueError("a macro average requires every declared group")
    k = len(values)
    return Estimate(
        sum((value.value for value in values), Fraction()) / k,
        sum((value.variance for value in values), Fraction()) / (k * k),
        sum(value.n for value in values),
    )
