"""Build results/summary.csv (one row per configuration, used for plotting) and
results/summary_table.csv (one row per activation, for reading) from every
per-epoch CSV in results/raw/.

Test accuracy at epoch E is taken from val_accuracy (runs trained with
validation_data=(test_images, test_labels)) or, for runs without it, from the
final test_accuracy when E is the last epoch.
Files whose name ends in _oldrun.csv are only used where no newer run exists;
when both exist, the two are printed side by side as a reproducibility check.

Usage:  python tools/make_summary.py
"""
import csv, glob, os, statistics as st
from collections import defaultdict

LAYERS = [1, 5, 10]
EPOCHS = [5, 10]
ACTS = ["ReLU", "Sigmoid", "tanh", "LeakyReLU", "GELU"]
ROOT = os.path.join(os.path.dirname(__file__), "..", "results")

def collect(path):
    out = defaultdict(dict)   # (layers, epoch, act) -> {seed: acc}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        e, total = int(r["epoch"]), int(r["total_epochs"])
        if e not in EPOCHS:
            continue
        if r["val_accuracy"] != "":
            acc = float(r["val_accuracy"])
        elif e == total:
            acc = float(r["test_accuracy"])
        else:
            continue
        out[(int(r["layers"]), e, r["activation"])][int(r["seed"])] = acc
    return out

new, old = {}, {}
for p in sorted(glob.glob(os.path.join(ROOT, "raw", "*.csv"))):
    target = old if p.endswith("_oldrun.csv") else new
    for key, seeds in collect(p).items():
        target[key] = (seeds, os.path.basename(p))

rows, missing = [], []
for E in EPOCHS:
    for L in LAYERS:
        for A in ACTS:
            key = (L, E, A)
            src = new.get(key) or old.get(key)
            if key in new and key in old:
                print(f"check {L}L/{E}E {A}: new {sorted(new[key][0].values())}  old {sorted(old[key][0].values())}")
            row = {"layers": L, "epochs": E, "activation": A,
                   "seed_1": "", "seed_2": "", "seed_3": "", "mean": "", "std": ""}
            if src:
                seeds = src[0]
                accs = [seeds[s] for s in sorted(seeds)]
                for i, s in enumerate(sorted(seeds)[:3]):
                    row[f"seed_{i+1}"] = f"{seeds[s]:.4f}"
                row["mean"] = f"{st.mean(accs):.4f}"
                row["std"] = f"{st.stdev(accs):.4f}" if len(accs) > 1 else ""
            else:
                missing.append(f"{L}L/{E}E {A}")
            rows.append(row)

out = os.path.join(ROOT, "summary.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"{out}: {len(rows) - len(missing)}/{len(rows)} cells done")

# Human-readable version: one row per activation, one column per configuration.
wide_out = os.path.join(ROOT, "summary_table.csv")
cols = [(L, E) for E in EPOCHS for L in LAYERS]
# utf-8-sig adds a BOM so Excel shows "±" correctly when the file is double-clicked
with open(wide_out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["activation"] + [f"{L} layer{'s' if L > 1 else ''} / {E} epochs" for L, E in cols])
    for A in ACTS:
        cells = []
        for L, E in cols:
            r = next(r for r in rows if r["layers"] == L and r["epochs"] == E and r["activation"] == A)
            cells.append(f"{float(r['mean']):.3f} ± {float(r['std']):.3f}" if r["mean"] else "")
        w.writerow([A] + cells)
print(f"{wide_out}: human-readable table")
if missing:
    print("missing:", ", ".join(missing))
