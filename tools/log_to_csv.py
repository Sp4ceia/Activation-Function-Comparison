"""Convert a Keras training log (copied from the notebook output) into a per-epoch CSV.

Usage:
    python tools/log_to_csv.py LOG.txt results/raw/OUT.csv [--layers N --neurons N --batch N]

The log is expected to contain, for every activation function:
    Activation function of the currently trained model:  ReLU
    Epoch 1/10
    391/391 - 3s - 7ms/step - accuracy: 0.79 - loss: 0.58 [- val_accuracy: ... - val_loss: ...]
    ...
    ReLU 1 0.8611999750137329          <- print(name, seed, test_acc)

If the log has a line like
    CONFIG layers=5 neurons=128 epochs=10 batch_size=128
those values are used; otherwise pass --layers / --neurons / --batch.
"""
import argparse, csv, re, sys
from collections import defaultdict

NAME_FIX = {"Gelu": "GELU"}

def parse(path, layers=None, neurons=None, batch=None):
    cfg = {"layers": layers, "neurons": neurons, "batch_size": batch}
    rows, buf, act, epoch, total = [], [], None, None, None
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        m = re.match(r"CONFIG\s+(.*)", line)
        if m:
            for k, v in re.findall(r"(\w+)=(\d+)", m.group(1)):
                if k in ("layers", "neurons", "batch_size"):
                    cfg[k] = int(v)
            continue
        m = re.match(r"Activation function of the currently trained model:\s+(\S+)", line)
        if m:
            if buf:
                sys.exit(f"ERROR: run of {act} has epochs but no result line (log cut?)")
            act = m.group(1)
            continue
        m = re.match(r"Epoch (\d+)/(\d+)$", line)
        if m:
            epoch, total = int(m.group(1)), int(m.group(2))
            continue
        if "accuracy:" in line and "/step" in line:
            vals = dict((k, float(v)) for k, v in re.findall(r"(\w+): ([0-9.eE+-]+)", line))
            buf.append((epoch, vals))
            continue
        m = re.match(r"(\S+) (\d+) ([0-9.]+)$", line)
        if m:
            name, seed, test = m.group(1), int(m.group(2)), float(m.group(3))
            if name != act:
                sys.exit(f"ERROR: result line for {name} but current activation is {act}")
            if [e for e, _ in buf] != list(range(1, total + 1)):
                sys.exit(f"ERROR: {name} seed {seed} has epochs {[e for e, _ in buf]}, expected 1..{total}")
            for e, v in buf:
                rows.append({
                    "layers": cfg["layers"], "neurons": cfg["neurons"], "total_epochs": total,
                    "batch_size": cfg["batch_size"], "activation": NAME_FIX.get(name, name),
                    "seed": seed, "epoch": e,
                    "train_accuracy": v.get("accuracy"), "train_loss": v.get("loss"),
                    "val_accuracy": v.get("val_accuracy", ""), "val_loss": v.get("val_loss", ""),
                    "test_accuracy": round(test, 4),
                })
            buf = []
    if buf:
        print(f"WARNING: log ends in the middle of a {act} run - it was probably cut off when copying.")
    if None in cfg.values():
        sys.exit(f"ERROR: missing config {cfg}; add a CONFIG line or pass --layers/--neurons/--batch")
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log"); ap.add_argument("out")
    ap.add_argument("--layers", type=int); ap.add_argument("--neurons", type=int); ap.add_argument("--batch", type=int)
    a = ap.parse_args()
    rows = parse(a.log, a.layers, a.neurons, a.batch)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    runs = defaultdict(set)
    for r in rows:
        runs[r["activation"]].add(r["seed"])
    print(f"{len(rows)} rows -> {a.out}")
    for k, s in runs.items():
        print(f"  {k}: seeds {sorted(s)}")
    if len({len(s) for s in runs.values()}) > 1:
        print("WARNING: not every activation has the same number of seeds - check the log is complete.")

if __name__ == "__main__":
    main()
