"""Build every number, table and figure of the preprint from stored result files.

Inputs (all committed): reports/phase3/benchmark/summary.json (pre-registered benchmark),
reports/phase3/theory/{union_limit, union_predict_benchmark, union_kappa, rhohat_tuning}.json,
reports/phase3/posthoc/summary.json (post hoc comparators), reports/phase3/danish.json.
Outputs: paper/preprint/generated/*.tex (macros and tables), paper/preprint/figures/*.pdf.
No number in the paper is typed by hand: the text uses the macros in numbers.tex.

Usage: python scripts/build_preprint_assets.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "phase3"
OUT = ROOT / "paper" / "preprint"
GEN = OUT / "generated"
FIG = OUT / "figures"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)
FAM = {"burr": "Burr", "frechet": "Fr\\'echet", "abs_t": "$|t|$"}
FAMP = {"burr": "Burr", "frechet": "Fréchet", "abs_t": "|t|"}


def load(p):
    return json.loads(Path(p).read_text())


def f3(x):
    return f"{x:.3f}"


def rng_str(v, fmt=f3):
    lo, hi = fmt(min(v)), fmt(max(v))
    return lo if lo == hi else f"{lo}--{hi}"


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    M: dict[str, str] = {}

    # ---------- pre-registered benchmark ----------
    bench = load(REP / "benchmark" / "summary.json")
    rows = bench["rows"]
    dec = bench["decisions"]
    ina = [r for r in rows if r["in_assumption"]]
    out = [r for r in rows if not r["in_assumption"]]
    M["nCellsIn"] = str(len(ina))
    M["benchReps"] = str(rows[0]["reps"])
    M["seBench"] = f"{math.sqrt(0.09 / rows[0]['reps']):.3f}"
    M["DoneA"] = str(dec["D1a_rhohat_band_after_selection_ge_0.85_all_rules"])
    M["DoneB"] = str(dec["D1b_union_band_after_selection_ge_0.85_all_rules"])
    M["Dtwo"] = str(dec["D2_first_order_lepski_interval_lt_0.80"])
    M["Dthree"] = str(dec["D3_bayes_band_closer_to_nominal_than_wald"])
    us = [r["epd_union"]["simultaneous"] for r in ina]
    M["unionSimIn"] = rng_str(us)
    ups = [min(v["band"] for v in r["epd_union"]["after"].values()) for r in ina]
    M["unionPostMinIn"] = f3(min(ups))
    fo = [r["pareto"]["after"]["lepski"]["interval"] for r in ina]
    fo_low = [v for v in fo if v < 0.8]
    M["foLepskiN"] = str(len(fo_low))
    M["foLepskiRange"] = rng_str(fo_low)
    M["foLepskiMin"] = f"{min(fo):.2f}"
    uo = [r["epd_union"]["simultaneous"] for r in out]
    M["unionSimOut"] = rng_str(uo)
    ratio = [r["epd_union"]["width90_top"] / r["epd_rhohat"]["width90_top"] for r in ina]
    M["unionVsPluginWidth"] = f"{min(ratio):.1f}--{max(ratio):.1f}"
    gps = [r["gp"]["simultaneous"] for r in ina]
    M["gpSimIn"] = rng_str(gps)

    lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r" & & & & \multicolumn{2}{c}{Union band} & \multicolumn{2}{c}{First-order} & "
        r"\multicolumn{2}{c}{GP band}\\",
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r"Family & $\rho$ & $\xi$ & $n$ & sim. & min post-sel. & sim. & Lepski int. & "
        r"sim. & min post-sel.\\",
        r"\midrule",
    ]
    for r in sorted(
        rows, key=lambda r: (not r["in_assumption"], r["family"], r["rho"], r["xi"], r["n"])
    ):

        def mn(m, r=r):
            return min(v["band"] for v in r[m]["after"].values())

        star = "" if r["in_assumption"] else r"$^{\dagger}$"
        lines.append(
            f"{FAM[r['family']]}{star} & ${r['rho']:g}$ & {r['xi']:g} & {r['n']} & "
            f"{r['epd_union']['simultaneous']:.2f} & {mn('epd_union'):.2f} & "
            f"{r['pareto']['simultaneous']:.2f} & "
            f"{r['pareto']['after']['lepski']['interval']:.2f} & "
            f"{r['gp']['simultaneous']:.2f} & {mn('gp'):.2f}\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_bench.tex").write_text("\n".join(lines) + "\n")

    # ---------- limit theory ----------
    ul = load(REP / "theory" / "union_limit.json")
    lam = np.array(ul["lambda"])
    m6 = ul["meshes"]["6"]
    rws = m6["rows"]
    inR = [r for r in rws if -2 - 1e-9 <= r["rho"] <= -0.5 + 1e-9]
    worst = min(min(r["coverage"]) for r in inR)
    M["GammaMinR"] = f3(worst)
    M["nRhoEval"] = str(len(inR))
    M["lamMax"] = f"{int(max(lam))}"
    M["simsLimit"] = str(ul["sims"])
    M["seLimit"] = f"{math.sqrt(0.09 / ul['sims']):.3f}"
    i0 = int(np.where(lam == 0)[0][0])
    M["GammaZeroMin"] = f3(min(r["coverage"][i0] for r in inR))
    M["widthRatioZero"] = f"{max(r['width_top'][i0] / r['oracle_width_top'] for r in inR):.1f}"
    M["widthRatioMax"] = f"{max(max(r['width_top']) / r['oracle_width_top'] for r in inR):.1f}"
    for key, rho in (
        ("GammaOutA", -0.45),
        ("GammaOutB", -0.25),
        ("GammaOutC", -2.5),
        ("GammaOutD", -3.0),
    ):
        row = [r for r in rws if abs(r["rho"] - rho) < 1e-6][0]
        M[key] = f3(min(row["coverage"]))
    same = all(
        abs(
            min(
                min(r["coverage"])
                for r in ul["meshes"][k]["rows"]
                if -2 - 1e-9 <= r["rho"] <= -0.5 + 1e-9
            )
            - worst
        )
        < 0.005
        for k in ("12", "24")
    )
    M["meshInsensitive"] = "identical" if same else "different"
    M["limitKs"] = ", ".join(str(k) for k in ul["ks"])

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    sel = [-0.5, -0.758, -1.0, -1.516, -2.0]
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(sel)))
    for c, rho in zip(cols, sel, strict=True):
        row = min(rws, key=lambda r: abs(r["rho"] - rho))
        ax[0].plot(lam, row["coverage"], color=c, lw=1, label=f"$\\rho={row['rho']:.2f}$")
    for c, rho in zip(["0.4", "0.6", "0.8"], [-0.25, -2.5, -3.0], strict=True):
        row = [r for r in rws if abs(r["rho"] - rho) < 1e-6][0]
        ax[0].plot(
            lam, row["coverage"], color=c, lw=1, ls="--", label=f"$\\rho={rho:g}$ ($\\notin R$)"
        )
    ax[0].axhline(0.9, color="k", lw=0.6, ls=":")
    ax[0].set(
        xlabel=r"$\lambda/\xi$ at $k_{\max}$",
        ylabel=r"limiting coverage $\Gamma_M$",
        ylim=(0, 1.02),
        xlim=(-48, 48),
    )
    ax[0].legend(fontsize=6.5, ncol=2, loc="lower center")
    for c, rho in zip(cols, sel, strict=True):
        row = min(rws, key=lambda r: abs(r["rho"] - rho))
        ax[1].plot(lam, np.array(row["width_top"]) / row["oracle_width_top"], color=c, lw=1)
    ax[1].axhline(1, color="k", lw=0.6, ls=":")
    ax[1].set(
        xlabel=r"$\lambda/\xi$ at $k_{\max}$", ylabel="width / known-$\\rho$ width", xlim=(-48, 48)
    )
    fig.tight_layout()
    fig.savefig(FIG / "limit_coverage_width.pdf")
    plt.close(fig)

    # ---------- predicted vs observed ----------
    pr = load(REP / "theory" / "union_predict_benchmark.json")
    pin = [r for r in pr if r["in_assumption"]]
    d = [r["observed"] - r["predicted"] for r in pin]
    M["predIn"] = rng_str([r["predicted"] for r in pin])
    M["obsIn"] = rng_str([r["observed"] for r in pin])
    M["diffIn"] = f"\\ensuremath{{{min(d):+.3f}}} to \\ensuremath{{{max(d):+.3f}}}"
    M["diffAbsMax"] = f3(max(abs(x) for x in d))
    pout = [r for r in pr if not r["in_assumption"]]
    M["predOut"] = rng_str([r["predicted"] for r in pout])
    M["obsOut"] = rng_str([r["observed"] for r in pout])
    lammax = [abs(r["lambda_over_xi"][-1]) for r in pin]
    M["lamBenchMax"] = f"{max(lammax):.1f}"
    dr_in = [abs(r["lambda_over_xi"][-1]) / math.sqrt(r["ks"][-1]) for r in pin]
    dr_out = [abs(r["lambda_over_xi"][-1]) / math.sqrt(r["ks"][-1]) for r in pout]
    M["deltaRatioIn"] = f"{max(dr_in):.2f}"
    M["deltaRatioOut"] = rng_str(dr_out, lambda x: f"{x:.2f}")
    se = math.sqrt(0.09 / 300)
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    mk = {"burr": "o", "frechet": "s", "abs_t": "^"}
    for r in pr:
        ax.errorbar(
            r["predicted"],
            r["observed"],
            yerr=2 * se,
            fmt=mk[r["family"]],
            ms=3.5,
            mfc="white" if not r["in_assumption"] else "C0",
            mec="C0" if r["in_assumption"] else "C3",
            ecolor="0.6",
            elinewidth=0.6,
            capsize=0,
        )
    ax.plot([0, 1], [0, 1], "k:", lw=0.6)
    ax.set(
        xlim=(0.84, 1.0),
        ylim=(0.1, 1.01),
        xlabel="predicted (limit theory)",
        ylabel="observed (pre-registered benchmark)",
    )
    for fam, m in mk.items():
        ax.plot([], [], m, color="C0", ms=3.5, label=FAMP[fam])
    ax.plot([], [], "o", mfc="white", mec="C3", ms=3.5, label=r"Burr $\rho=-0.25$")
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "predicted_vs_observed.pdf")
    plt.close(fig)

    # ---------- uniform endpoint bound (Theorem uniform, Corollary uniform) ----------
    eb = load(REP / "theory" / "union_endpoint_bound.json")
    two, one = eb["two_sided"], eb["one_sided_endpoints"]
    M["nEndpointDraws"] = f"{eb['gaussian_draws']:,}".replace(",", "{,}")
    M["cTwoEnd"] = rng_str([two["c"][0], two["c"][-1]], lambda x: f"{x:.3f}")
    M["cOneEnd"] = rng_str([one["c"][0], one["c"][-1]], lambda x: f"{x:.3f}")
    pis = [two["pi_up"][0], two["pi_up"][-1], two["pi_down"][0], two["pi_down"][-1]]
    M["piTwoMax"] = f"{max(pis):.4f}"
    M["qMax"] = f"{max(two['q'][0], two['q'][-1]):.5f}"
    phi2 = max(two["phi_plus"], two["phi_minus"])
    M["boundTwo"] = f3(1 - phi2)
    M["upperTwo"] = f3(1 - max(pis))
    M["GminOne"] = f3(one["Gamma_min"])
    M["widthOneSided"] = f"{100 * (max(one['width_factor_endpoints']) - 1):.1f}"
    fin = eb["finite_k_pareto"]
    M["finUp"] = f3(fin["two_sided"]["pi_up"][0])
    M["finDn"] = f3(fin["two_sided"]["pi_down"][0])
    fc2, fc1 = fin["two_sided"]["c"], fin["one_sided_endpoints"]["c"]
    M["finOneFactor"] = f"{100 * max(fc1[0] / fc2[0], fc1[-1] / fc2[-1]) - 100:.0f}"

    # ---------- kappa ----------
    kap = load(REP / "theory" / "union_kappa.json")
    k90 = [k for k in kap if abs(k["kappa"] - 0.9) < 1e-9][0]
    M["kappaWorst"] = f3(k90["worst_coverage"])

    # ---------- rho-hat audit ----------
    rt = load(REP / "theory" / "rhohat_tuning.json")

    def q(fam, rho, n, key):
        r = [x for x in rt if x["family"] == fam and abs(x["rho"] - rho) < 1e-9 and x["n"] == n][0]
        return r["est"][key]

    M["rhoNineNineNineBurrA"] = f"{q('burr', -0.25, 1_000_000, '0.999|0')[1]:.2f}"
    M["rhoNineNineNineBurrB"] = f"{q('burr', -0.5, 1_000_000, '0.999|0')[1]:.2f}"
    M["rhoNineNineNineBurrC"] = f"{q('burr', -1.0, 1_000_000, '0.999|0')[1]:.2f}"
    M["rhoNineBurrC"] = f"{q('burr', -1.0, 1_000_000, '0.9|0')[1]:.2f}"
    M["rhoNineBurrB"] = f"{q('burr', -0.5, 1_000_000, '0.9|0')[1]:.2f}"
    M["rhoNineBurrA"] = f"{q('burr', -0.25, 1_000_000, '0.9|0')[1]:.2f}"
    M["rhoNineFrechet"] = f"{q('frechet', -1.0, 1_000_000, '0.9|0')[1]:.2f}"
    z = q("burr", -1.0, 5000, "0.9|0")
    M["rhoNineBurrCsmall"] = (
        f"\\ensuremath{{{z[1]:.2f}}} (interquartile range "
        f"\\ensuremath{{{z[0]:.2f}}} to \\ensuremath{{{z[2]:.2f}}})"
    )
    M["rhoFrechetSwitch"] = f"{q('frechet', -1.0, 1_000_000, '0.999|1')[1]:.2f}"

    # ---------- post hoc comparators ----------
    ph_path = REP / "posthoc" / "summary.json"
    if ph_path.exists():
        ph = load(ph_path)
        key = {(r["family"], r["rho"], r["xi"], r["n"]): r for r in rows}
        L = [
            r"\begin{tabular}{llrrrrrrrrr}",
            r"\toprule",
            r" & & & & \multicolumn{3}{c}{Sasaki--Wang $H^{O}$ (pointwise)} & "
            r"\multicolumn{2}{c}{CGP plug-in band} & \multicolumn{2}{c}{Union band}\\",
            r"\cmidrule(lr){5-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
            r"Family & $\rho$ & $\xi$ & $n$ & cov.\ $k_{\max}$ & after GH & width $k_{\max}$ & "
            r"sim. & width $k_{\max}$ & sim. & width $k_{\max}$\\",
            r"\midrule",
        ]
        sw_top, sw_gh, cgp_sim = [], [], []
        for r in sorted(
            ph, key=lambda r: (not r["in_assumption"], r["family"], r["rho"], r["xi"], r["n"])
        ):
            b = key[(r["family"], r["rho"], r["xi"], r["n"])]
            star = "" if r["in_assumption"] else r"$^{\dagger}$"
            L.append(
                f"{FAM[r['family']]}{star} & ${r['rho']:g}$ & {r['xi']:g} & {r['n']} & "
                f"{r['sw_pointwise'][-1]:.2f} & {r['sw_after_guillou_hall']:.2f} & "
                f"{r['sw_width'][-1]:.3f} & {r['cgp_post_simultaneous']:.2f} & "
                f"{r['cgp_post_bandwidth_top']:.3f} & {b['epd_union']['simultaneous']:.2f} & "
                f"{b['epd_union']['widthband_top']:.3f}\\\\"
            )
            if r["in_assumption"]:
                sw_top.append(r["sw_pointwise"][-1])
                sw_gh.append(r["sw_after_guillou_hall"])
                cgp_sim.append(r["cgp_post_simultaneous"])
        L += [r"\bottomrule", r"\end{tabular}"]
        (GEN / "tab_posthoc.tex").write_text("\n".join(L) + "\n")
        M["swTopIn"] = rng_str(sw_top, lambda x: f"{x:.2f}")
        good = [
            r["sw_pointwise"][-1]
            for r in ph
            if r["in_assumption"] and not (r["family"] == "burr" and r["rho"] == -0.5)
        ]
        bad = [r["sw_pointwise"][-1] for r in ph if r["family"] == "burr" and r["rho"] == -0.5]
        M["swTopGoodMin"] = f"{min(good):.2f}"
        M["swTopBurr"] = rng_str(bad, lambda x: f"{x:.2f}")
        wr = [
            r["sw_width"][-1]
            / key[(r["family"], r["rho"], r["xi"], r["n"])]["epd_union"]["widthband_top"]
            for r in ph
            if r["in_assumption"]
        ]
        M["swWidthRatio"] = rng_str(wr, lambda x: f"{x:.2f}")
        M["swGHIn"] = rng_str(sw_gh, lambda x: f"{x:.2f}")
        M["cgpSimIn"] = rng_str(cgp_sim, lambda x: f"{x:.2f}")
        M["cgpSimInN"] = str(sum(v >= 0.85 for v in cgp_sim))
        M["posthocAvailable"] = "1"
        sn = load(REP / "posthoc" / "sw_snoop_summary.json")
        sn_in = [r for r in sn if r["in_assumption"]]
        g = [
            r["sw_snoop_coverage"]
            for r in sn_in
            if not (r["family"] == "burr" and r["rho"] == -0.5)
        ]
        bb = [r["sw_snoop_coverage"] for r in sn_in if r["family"] == "burr" and r["rho"] == -0.5]
        M["swSnoopGood"] = rng_str(g, lambda x: f"{x:.2f}")
        M["swSnoopBurr"] = rng_str(bb, lambda x: f"{x:.2f}")
        M["swSnoopOut"] = rng_str(
            [r["sw_snoop_coverage"] for r in sn if not r["in_assumption"]], lambda x: f"{x:.2f}"
        )
        wr2 = [r["sw_snoop_width_median"] / r["union_intersection_width_median"] for r in sn_in]
        M["swSnoopWidthRatio"] = rng_str(wr2, lambda x: f"{x:.2f}")
    else:
        M["posthocAvailable"] = "0"

    # ---------- Danish ----------
    dn = load(REP / "danish.json")
    ks = dn["ks"]
    M["danishN"] = str(dn["n"])
    M["danishKs"] = ", ".join(map(str, ks))
    M["danishSamsee"] = str(dn["selector_k_raw"]["samsee"])
    M["danishGH"] = str(dn["selector_k_raw"]["guillou_hall"])
    U = np.array(dn["methods"]["epd_union"]["iv"])
    sel = dn["methods"]["epd_union"]["sel"]
    M["danishKsam"] = str(ks[sel["samsee"]])
    M["danishKgh"] = str(ks[sel["guillou_hall"]])
    inter = (U[:, 3].max(), U[:, 4].min())
    M["danishInter"] = f"[{inter[0]:.3f}, {inter[1]:.3f}]"
    L = [
        r"\begin{tabular}{rcccc}",
        r"\toprule",
        r"$k$ & Union band & First-order band & GP band & First-order 90\% int.\\",
        r"\midrule",
    ]
    P = np.array(dn["methods"]["pareto"]["iv"])
    G = np.array(dn["methods"]["gp"]["iv"])
    for j, k in enumerate(ks):
        L.append(
            f"{k} & [{U[j, 3]:.3f}, {U[j, 4]:.3f}] & [{P[j, 3]:.3f}, {P[j, 4]:.3f}] & "
            f"[{G[j, 3]:.3f}, {G[j, 4]:.3f}] & [{P[j, 0]:.3f}, {P[j, 2]:.3f}]\\\\"
        )
    L += [
        r"\midrule",
        f"$\\cap_j$ & [{inter[0]:.3f}, {inter[1]:.3f}] & & & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (GEN / "tab_danish.tex").write_text("\n".join(L) + "\n")
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    x = np.arange(len(ks))
    for off, arr, lab, c in (
        (-0.2, U, "union band", "C0"),
        (0.0, G, "GP band", "C2"),
        (0.2, P, "first-order band", "C3"),
    ):
        ax.vlines(x + off, arr[:, 3], arr[:, 4], color=c, lw=2, label=lab)
        ax.plot(x + off, arr[:, 1], "o", color=c, ms=2.5)
    ax.axhspan(*inter, color="C0", alpha=0.12, lw=0, label=r"$\cap_j U_j$")
    ax.set_xticks(x, [str(k) for k in ks])
    ylim = (0.2, 1.7)
    clipped = [
        (k, a[4]) for arr in (U, G, P) for k, a in zip(ks, arr, strict=True) if a[4] > ylim[1]
    ]
    ax.set(xlabel="$k$", ylabel=r"$\xi$", ylim=ylim)
    M["danishClip"] = "; ".join(f"$k={k}$: upper end {v:.2f}" for k, v in clipped) or "none"
    ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "danish.pdf")
    plt.close(fig)

    for k in list(M):
        if k.startswith("rho") and k != "rhoNineBurrCsmall":
            M[k] = f"\\ensuremath{{{M[k]}}}"
    tex = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items()))
    (GEN / "numbers.tex").write_text(
        "% generated by scripts/build_preprint_assets.py\n" + tex + "\n"
    )
    print(tex)


if __name__ == "__main__":
    main()
