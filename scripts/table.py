"""Print the raw grid: excess over the exact floor (nT^2) as size x D per condition, for the
main grid, the small-D block, the fixed-mask study and the seed repeats. Read this before fitting."""
import glob
import json
import re
from collections import defaultdict

SIZES = ["xxs", "xs", "s", "m", "l", "xl", "xxl"]
pat = re.compile(r"^vit(xxs|xs|s|m|l|xl|xxl)_b([\d.]+)_h(\d+)_D(\d+)_s(\d+)(_fm)?\.json$")

cells = defaultdict(dict)      # (beta, h, tag) -> {(size, D): [excess per seed]}
for f in sorted(glob.glob("runs/*.json")):
    m = pat.match(f.split("/")[-1])
    if not m:
        continue
    sz, b, h, D, seed, tag = m.groups()
    r = json.load(open(f))
    hk = str(float(h))
    ex = r["best_val"][hk] - r["floor"][hk]
    cells[(b, h, tag or "")].setdefault((sz, int(D)), []).append(ex)

for (b, h, tag), grid in sorted(cells.items()):
    Ds = sorted({D for _, D in grid})
    print(f"\nbeta={b} h={h} m {'FIXED MASK' if tag else ''}   excess over floor [nT^2], (n seeds) where > 1")
    print("size   " + "".join(f"{D:>12d}" for D in Ds))
    for sz in SIZES:
        row = []
        for D in Ds:
            v = grid.get((sz, D))
            if v is None:
                row.append(f"{'':>12}")
            elif len(v) == 1:
                row.append(f"{v[0]:12.3f}")
            else:
                row.append(f"{sum(v)/len(v):9.3f}({len(v)})")
        if any(c.strip() for c in row):
            print(f"{sz:6s} " + "".join(row))
