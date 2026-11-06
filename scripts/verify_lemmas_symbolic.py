"""Computer-algebra check of the identities used in the preprint appendix (Lemma A1).

Checks, with E ~ Exp(1) and ξ = 1 without loss of generality:
  (i)   EPD(ρ′) δ-score at δ = 0 equals a_{ρ′}(E) = (1 − ρ′)e^{ρ′E} − 1;
  (ii)  efficient influence ψ_{ρ′} has variance ((1 − ρ′)/ρ′)²;
  (iii) the second-order direction implied by (A1) equals a_ρ(E);
  (iv)  cov(ψ_{ρ′}, a_ρ) = β(ρ, ρ′);
  (v)   the BJS (2009) estimator's influence function equals ψ_{ρ′}.
Exits non-zero if any check fails. Requires sympy.
"""

from __future__ import annotations

import sys

import sympy as sp


def main() -> int:
    r, rp, E, xi, d = sp.symbols("rho rhop E xi delta")
    c = rp / (1 - rp)
    dd = rp**2 / (1 - 2 * rp)
    A1 = sp.simplify(dd / (dd - c**2))
    A2 = sp.simplify(-c / (dd - c**2))
    ok = {}
    ok["psi coefficients"] = (
        sp.simplify(A1 - (1 - rp) ** 2 / rp**2) == 0
        and sp.simplify(A2 + (1 - 2 * rp) * (1 - rp) / rp**3) == 0
    )
    ok["variance"] = sp.simplify(A1 - ((1 - rp) / rp) ** 2) == 0
    y = sp.exp(xi * E)
    surv = (y * (1 + d - d * sp.exp(rp * E))) ** (-1 / xi)  # y^{ρ′/ξ} = e^{ρ′E} exactly
    dens = -sp.diff(surv, E)
    score = sp.diff(sp.log(dens), d).subs(d, 0)
    ok["EPD delta-score"] = (
        sp.simplify(sp.powsimp(sp.expand(score * xi), force=True) - ((1 - rp) * sp.exp(rp * E) - 1))
        == 0
    )
    s2 = sp.exp(-E) * (1 + d * (sp.exp(r * E) - 1))
    f2 = -sp.diff(s2, E)
    ok["second-order direction"] = (
        sp.simplify(sp.diff(sp.log(f2), d).subs(d, 0) - ((1 - r) * sp.exp(r * E) - 1)) == 0
    )
    w1 = r / (1 - r)
    w2 = r * rp / (1 - r - rp)
    beta = r * (1 - rp) * (rp - r) / (rp * (1 - r) * (1 - r - rp))
    ok["beta"] = sp.simplify(A1 * w1 + A2 * w2 - beta) == 0
    # BJS influence (ξ = 1): ξ̂ = H − δ̂ρ′/(1−ρ′), δ̂ = H(1−2ρ′)(1−ρ′)³ρ′⁻⁴(M − 1/(1−ρ′)),
    # M = mean exp(ρ′E/H). Influence of M: (e^{ρ′E} − 1/(1−ρ′)) − ρ′E[E e^{ρ′E}](E − 1).
    C = (1 - 2 * rp) * (1 - rp) ** 3 / rp**4
    coef_E = 1 - rp / (1 - rp) * C * (-rp / (1 - rp) ** 2)
    coef_e = -rp / (1 - rp) * C
    ok["BJS = efficient"] = (
        sp.simplify(coef_E - A1) == 0 and sp.simplify(coef_e - A2 * (1 - rp)) == 0
    )
    x = sp.Symbol("x", positive=True)
    num = {
        "cov(E-1,a)": (
            (x - 1) * ((1 - sp.Rational(-1, 2)) * sp.exp(sp.Rational(-1, 2) * x) - 1),
            sp.Rational(-1, 2) / sp.Rational(3, 2),
        )
    }
    for name, (expr, target) in num.items():
        ok[name] = sp.simplify(sp.integrate(expr * sp.exp(-x), (x, 0, sp.oo)) - target) == 0
    for k, v in ok.items():
        print(f"{k}: {'ok' if v else 'FAILED'}")
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
