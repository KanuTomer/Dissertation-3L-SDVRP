# alpha_sweep.py
import subprocess, shlex, csv, os
MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
alphas = [1e6, 1e7, 1e8]
out = []
for a in alphas:
    cmd = f'python route_ga.py --merged "{MERGED}" --route_size 8 --pop 40 --gens 120 --alpha {a} --seed 42'
    print("Running:", cmd)
    subprocess.run(shlex.split(cmd), check=True)
    csvpath = os.path.join("route_ga_out","route_ga_summary.csv")
    if os.path.exists(csvpath):
        with open(csvpath) as f:
            r = list(csv.reader(f))
            out.append((a, r[1]))
    else:
        out.append((a, ["no output"]))
print("\nSweep results:")
for row in out:
    print(row)
