#!/usr/bin/env python3
# quick_inspect.py
# Quick wrapper to run inspect_order_any.py for a given route file and print a short summary
# Usage: python quick_inspect.py path/to/route.json 38

import sys
import subprocess
from pathlib import Path
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_inspect.py path/to/route.json [route_id]")
        return
    route = Path(sys.argv[1])
    if not route.exists():
        print("File not found:", route)
        return
    route_id = sys.argv[2] if len(sys.argv) > 2 else None
    # If inspect_order_any.py exists, call it. Otherwise print quick summary from JSON
    if Path('inspect_order_any.py').exists():
        cmd = ['python','inspect_order_any.py', str(route)]
        if route_id:
            cmd.append(route_id)
        subprocess.run(cmd)
        return
    # Fallback: open JSON and show high-level keys
    data = json.load(open(route))
    print("Top-level keys:", list(data.keys()))
    # print a few entries if present
    if isinstance(data, dict):
        for k in list(data.keys())[:10]:
            print(k, "-> type:", type(data[k]).__name__)
    elif isinstance(data, list):
        print("List length:", len(data))
        if data:
            print("First item keys:", list(data[0].keys()))

if __name__ == '__main__':
    main()
