
# compare_metrics.py
# Comparative study for metric laws (loi1, loi2, loi3) using helper functions
# Requires the environment where PhysParams, AdaptParams, study_NX_vs_err, solve_adrs_adapt are defined.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def run_comparisons(PhysParams, AdaptParams, study_NX_vs_err, solve_adrs_adapt):
    phys = PhysParams()
    err_values = [0.04, 0.02, 0.01, 0.005, 0.0025]
    laws = ["loi1", "loi2", "loi3"]
    labels_map = {"loi1": "Loi 1 (finale)", "loi2": "Loi 2 (moyenne temp.)", "loi3": "Loi 3 (RMS temp.)"}

    ap_template = AdaptParams(
        hmin=0.01, hmax=0.25, err=0.01, NX_init=12,
        NT_max=400, niter_refinement=6, NX_tol=1,
        L2_target=None, metric_law="loi3"
    )

    # NX(err)
    df_list = []
    slopes_info = {}

    for law in laws:
        ap_template.metric_law = law
        df = study_NX_vs_err(err_values, phys, ap_template)
        df.rename(columns={"NX_final": f"NX_{law}"}, inplace=True)
        df_list.append(df)
        log_err = np.log(np.array(df["err"]))
        log_NX = np.log(np.array(df[f"NX_{law}"]))
        A = np.vstack([log_err, np.ones_like(log_err)]).T
        slope, intercept = np.linalg.lstsq(A, log_NX, rcond=None)[0]
        slopes_info[law] = slope

    df_merged = df_list[0]
    for d in df_list[1:]:
        df_merged = df_merged.merge(d, on="err", how="inner")

    out_csv_compare = Path("compare_NX_vs_err.csv")
    df_merged.to_csv(out_csv_compare, index=False)

    plt.figure()
    for law in laws:
        plt.plot(df_merged["err"], df_merged[f"NX_{law}"], marker='o', label=f"{labels_map[law]} (pente ~ {abs(slopes_info[law]):.2f})")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("err (log)")
    plt.ylabel("NX_final (log)")
    plt.title("Comparaison NX(err) — lois de métrique")
    plt.legend()
    plt.savefig("NX_vs_err_compare.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Contraction
    ik_data = {}
    for law in laws:
        ap_c = AdaptParams(
            hmin=0.01, hmax=0.25, err=0.01, NX_init=12,
            NT_max=400, niter_refinement=8, NX_tol=1,
            L2_target=None, metric_law=law
        )
        x, T, hist = solve_adrs_adapt(phys, ap_c, return_history=True)
        ik_data[law] = np.array(hist["Ik"], dtype=float)

    max_len = max(len(ik_data[l]) for l in laws)
    df_contr = pd.DataFrame({"iter": np.arange(1, max_len+1)})
    for l in laws:
        arr = ik_data[l]
        if len(arr) < max_len:
            arr = np.concatenate([arr, np.full(max_len-len(arr), np.nan)])
        df_contr[f"Ik_{l}"] = arr

    df_contr.to_csv("contraction_compare.csv", index=False)

    plt.figure()
    for l in laws:
        plt.plot(df_contr["iter"], df_contr[f"Ik_{l}"], marker='o', label=labels_map[l])
    plt.xlabel("Itération d'adaptation k")
    plt.ylabel("||YBk - YBk-1|| (maillage background, L2)")
    plt.title("Contraction des solutions interpolées — comparaison des lois")
    plt.legend()
    plt.savefig("contraction_compare.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

if __name__ == "__main__":
    print("This module expects to be run in the environment where the helper functions are defined.")
