#!/usr/bin/env python3
import json, ast, argparse, pathlib, sys, re
import pandas as pd

def try_load_json(path):
    text = path.read_text(encoding="utf8")
    try:
        return json.loads(text)
    except:
        pass
    try:
        return ast.literal_eval(text)
    except:
        pass
    txt2 = text.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
    try:
        return json.loads(txt2)
    except:
        pass
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1:
        try:
            return json.loads(text[s:e+1])
        except:
            pass
    raise ValueError(f"Unable to parse JSON-like file: {path}")

def collect_results(run_dir):
    run_dir = pathlib.Path(run_dir)
    matches = list(run_dir.rglob("result_repaired.json")) + list(run_dir.rglob("result.json"))
    unique = {}
    for p in matches:
        folder = p.parent.resolve()
        if folder not in unique or p.name == "result_repaired.json":
            unique[folder] = p

    results, skipped = [], []
    for folder, p in sorted(unique.items()):
        try:
            data = try_load_json(p)
        except Exception as e:
            skipped.append((str(p), str(e)))
            continue

        if isinstance(data, list) and len(data)==1 and isinstance(data[0], dict):
            data = data[0]
        if not isinstance(data, dict):
            skipped.append((str(p), "parsed non-dict"))
            continue

        if "summary" in data and isinstance(data["summary"], dict):
            row = dict(data["summary"])
        else:
            row = dict(data)

        if "seed" not in row:
            m = re.search(r"seed[_-]?(\d+)", folder.name)
            if m:
                row["seed"] = int(m.group(1))

        results.append(row)

    return results, skipped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    results, skipped = collect_results(run_dir)

    print(f"Parsed: {len(results)} files, Skipped: {len(skipped)}")
    if skipped:
        print("Skipped examples:")
        for s in skipped[:10]:
            print("  ", s)

    if not results:
        print("No results found.")
        return

    df = pd.json_normalize(results)
    if "seed" not in df.columns:
        df["seed"] = df.index
    df = df.loc[:, ~df.columns.duplicated()]

    out = pathlib.Path(args.out) if args.out else run_dir / "replicates_summary.csv"
    df.to_csv(out, index=False)
    print("Summary written to:", out)

if __name__ == "__main__":
    main()
