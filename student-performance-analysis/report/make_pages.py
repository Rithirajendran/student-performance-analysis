"""Find the page of every heading / figure / table caption in the rendered PDF -> page_map.json"""
import json, subprocess, sys
pdf, items, out = sys.argv[1:4]
n = int([l for l in subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout.splitlines() if l.startswith("Pages")][0].split()[1])
pages = [subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "-layout", pdf, "-"], capture_output=True, text=True).stdout for i in range(1, n + 1)]
lines = [[l.strip() for l in p.splitlines()] for p in pages]
it = json.load(open(items)); pm = {}
body = max(i for i, L in enumerate(lines) if "1. Introduction and Objectives" in L)
for h in it["headings"]:
    hit = [i for i, L in enumerate(lines) if h in L and i >= body]
    if hit: pm[h] = str(hit[0] + 1)
    else: print("NOT FOUND heading:", h)
for c in it["figs"] + it["tabs"]:
    key = c[:28]
    hit = [i for i, L in enumerate(lines) if any(x.startswith(key) for x in L) and i >= body]
    if hit: pm[c] = str(hit[0] + 1)
    else: print("NOT FOUND caption:", c)
json.dump(pm, open(out, "w"), indent=1); print(len(pm), "entries; body starts on page", body + 1)
