"""Pick the best flat learning rate by validation KID, then queue its seed-3802 verification and
the architecture runs on it.

While the highest rate tried so far is also the best, step up the ladder (5e-4 -> 7.5e-4 -> 1e-3 -> 1.5e-3)
and wait for that run before deciding. Safe to re-run: runs already in sweep.txt are not queued
again."""
import json
import subprocess
import time
from pathlib import Path

root = Path("/workspace")
results_dir = root/"results"
sweep_file = root/"sweeps"/"sweep.txt"
rates = {"3e-4": 3e-4, "4e-4": 4e-4, "5e-4": 5e-4, "7.5e-4": 7.5e-4, "1e-3": 1e-3, "1.5e-3": 1.5e-3}
ladder = ["5e-4", "7.5e-4", "1e-3", "1.5e-3"]


def kid(name):
    results = results_dir/f"lab-05-hhtt85-lr-flat-{name}"/"lab-05-results.json"
    value = json.loads(results.read_text())["metrics"]["kid"]
    print(f"flat {name}: KID x 1000 = {1000*value:.3f}", flush=True)
    return value


def queue(lines):
    queued = {l.split(" ")[0] for l in sweep_file.read_text().splitlines() if l.strip()}
    lines = [l for l in lines if l.split(" ")[0] not in queued]
    if not lines:
        return
    with open(sweep_file, "a") as f:
        f.write("\n".join(lines) + "\n")
    print("queued:\n" + "\n".join(l.split(" ")[0] for l in lines), flush=True)
    # The sweep exits when it reaches the end of sweep.txt; restart it if so (it skips finished runs).
    time.sleep(5)
    status = subprocess.run(["supervisorctl", "status", "gan_sweep"], capture_output=True, text=True)
    if "RUNNING" not in status.stdout:
        subprocess.run(["supervisorctl", "start", "gan_sweep"])


def flat(name, seed=3801, suffix=""):
    lr = rates[name]
    return (f"lab-05-hhtt85-lr-flat-{name}{suffix} {seed} "
            f"{json.dumps({'learning_rate': lr, 'final_learning_rate': lr})}")


def wait_for(name):
    while not (results_dir/f"lab-05-hhtt85-lr-flat-{name}.zip").exists():
        time.sleep(15)


kids = {name: kid(name) for name in ("3e-4", "4e-4", "5e-4")}
winner = min(kids, key=kids.get)
print(f"winner so far: flat {winner}", flush=True)
while winner in ladder[:-1]:
    step = ladder[ladder.index(winner) + 1]
    queue([flat(step)])
    wait_for(step)
    kids[step] = kid(step)
    if min(kids, key=kids.get) != step:
        break
    winner = step
    print(f"winner so far: flat {winner}", flush=True)

lr = rates[winner]
print(f"winner: flat {winner}", flush=True)
lines = [flat(winner, seed=3802, suffix="-seed2")]
for line in (root/"sweeps"/"arch-template.txt").read_text().splitlines():
    out, seed, overrides = line.split(" ", 2)
    config = json.loads(overrides)
    config.update(learning_rate=lr, final_learning_rate=lr)
    lines.append(f"{out.replace('-lr-flat-3e-4', f'-lr-flat-{winner}')} {seed} "
                 f"{json.dumps(config, separators=(',', ':'))}")
queue(lines)
