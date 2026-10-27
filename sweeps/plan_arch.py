"""Once the architecture runs on flat 7.5e-4 have finished:

1. pick the best architecture on seed 3801 (including the vanilla architecture's own run);
2. run that winner without x-flips on seed 3801 and keep whichever of the two is better;
3. sweep the EMA half-life (2.5, 5, 20, 40 kimg, and 80 if 40 wins; default 10) on that and keep
   the best;
4. if the result isn't vanilla (x-flips, default EMA), queue it on seed 3802 so the final choice
   compares two-seed means against vanilla.

Safe to re-run: runs already in sweep.txt are not queued again."""
import json
import subprocess
import time
from pathlib import Path

root = Path("/workspace")
results_dir = root/"results"
sweep_file = root/"sweeps"/"sweep.txt"
variants = ["arch-g-heavy", "arch-deeper", "arch-shallow-wide", "arch-expand3", "arch-lowres-heavy",
            "arch-deep-gheavy"]
lr_tag = "lr-flat-7.5e-4"


def folder(name):
    stem = lr_tag if name == "vanilla" else f"{name}-{lr_tag}"
    return results_dir/f"lab-05-hhtt85-{stem}"


def results(path):
    return json.loads((path/"lab-05-results.json").read_text())


def kid(path):
    value = results(path)["metrics"]["kid"]
    print(f"{path.name:52s} KID x 1000 = {1000*value:.3f}", flush=True)
    return value


def wait_for(path):
    while not Path(f"{path}.zip").exists():
        time.sleep(15)


def queue(out, seed, config):
    overrides = {key: config[key] for key in
                 ("learning_rate", "final_learning_rate", "architecture", "xflip", "ema_kimg")}
    queued = {l.split(" ")[0] for l in sweep_file.read_text().splitlines() if l.strip()}
    if out not in queued:
        with open(sweep_file, "a") as f:
            f.write(f"{out} {seed} {json.dumps(overrides, separators=(',', ':'))}\n")
        print(f"queued: {out}", flush=True)
    # The sweep exits when it reaches the end of sweep.txt; restart it if so (it skips finished runs).
    time.sleep(5)
    status = subprocess.run(["supervisorctl", "status", "gan_sweep"], capture_output=True, text=True)
    if "RUNNING" not in status.stdout:
        subprocess.run(["supervisorctl", "start", "gan_sweep"])


for name in variants:
    wait_for(folder(name))
candidates = {name: folder(name) for name in ["vanilla"] + variants}
kids = {name: kid(path) for name, path in candidates.items()}
winner = min(kids, key=kids.get)
print(f"seed-3801 architecture winner: {winner}", flush=True)

best = folder(winner)
config = results(best)["config"]
no_flip = results_dir/f"{best.name}-noflip"
queue(no_flip.name, 3801, dict(config, xflip=False))
wait_for(no_flip)
if kid(no_flip) < kid(best):
    best, config = no_flip, dict(config, xflip=False)
print(f"seed-3801 winner: {best.name}", flush=True)

# EMA half-life sweep around the default 10 kimg, on the winner.
base = best
ema_runs = {kimg: results_dir/f"{base.name}-ema-{kimg:g}k" for kimg in (2.5, 5, 20, 40)}
for kimg, path in ema_runs.items():
    queue(path.name, 3801, dict(config, ema_kimg=kimg))
for kimg, path in ema_runs.items():
    wait_for(path)
    if kid(path) < kid(best):
        best, config = path, dict(config, ema_kimg=kimg)
if config["ema_kimg"] == 40:  # the longest half-life won: try one step longer
    path = results_dir/f"{base.name}-ema-80k"
    queue(path.name, 3801, dict(config, ema_kimg=80))
    wait_for(path)
    if kid(path) < kid(best):
        best, config = path, dict(config, ema_kimg=80)
print(f"seed-3801 winner after EMA sweep: {best.name} (ema_kimg {config['ema_kimg']:g})", flush=True)

if best == folder("vanilla"):
    print("vanilla with x-flips and the default EMA wins; nothing to verify", flush=True)
else:
    queue(f"{best.name}-seed2", 3802, config)
