#!/usr/bin/env python3
"""Print per-persona band-zero deltas."""
import json, glob, os
ROOT = "<REPO>"
files = sorted(glob.glob(f"{ROOT}/runs/32_band_zero/V3_P_*/summary.json"))
print(f"{'pid':<10} {'full_pres':>10} {'full_abs':>10} {'zero_pres':>10} {'zero_abs':>10} {'Δ_abs_pp':>10} {'Δ_pres_pp':>10}")
totals = {"f_pres":[], "f_abs":[], "z_pres":[], "z_abs":[]}
for fp in files:
    d = json.loads(open(fp).read())
    pid = d["pid"]
    full = d.get("C_lora_full", {}).get("absence", {})
    zero = d.get("C_lora_zero", {}).get("absence", {})
    if not full or not zero:
        continue
    fp_, fa, zp, za = full["present_tpr"], full["absence_tpr"], zero["present_tpr"], zero["absence_tpr"]
    totals["f_pres"].append(fp_); totals["f_abs"].append(fa)
    totals["z_pres"].append(zp); totals["z_abs"].append(za)
    print(f"{pid:<10} {fp_:>10.3f} {fa:>10.3f} {zp:>10.3f} {za:>10.3f} {100*(za-fa):>+10.1f} {100*(zp-fp_):>+10.1f}")
if totals["f_abs"]:
    n = len(totals["f_abs"])
    m = lambda xs: sum(xs)/len(xs)
    print(f"\n=== mean (n={n}) ===")
    print(f"  full: present_tpr={m(totals['f_pres']):.3f}  absence_tpr={m(totals['f_abs']):.3f}")
    print(f"  zero: present_tpr={m(totals['z_pres']):.3f}  absence_tpr={m(totals['z_abs']):.3f}")
    print(f"  Δ:    present_tpr={100*(m(totals['z_pres'])-m(totals['f_pres'])):+.1f}pp  absence_tpr={100*(m(totals['z_abs'])-m(totals['f_abs'])):+.1f}pp")
