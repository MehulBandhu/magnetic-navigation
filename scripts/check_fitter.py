"""Check the additive fitter recovers known exponents from synthetic excess-risk data.
usage: python scripts/check_fitter.py [noise]   noise = relative scatter on each cell (default 0.01).
At 0.03 scatter alpha comes back ~0.1 high; delta is robust. Alpha needs seeds."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from magscale.fit import fit_additive

A, B, alpha, delta = 300.0, 40.0, 0.45, 0.30
NOISE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
rng = np.random.default_rng(0)
N, D, L = [], [], []
for n in [1.3e5, 3.8e5, 1.3e6, 5e6, 1.5e7, 3.9e7]:
    for d in [2048, 16384, 131072, 1048576]:
        N.append(n); D.append(d); L.append((A * n ** -alpha + B * d ** -delta) * np.exp(NOISE * rng.standard_normal()))
fit, _ = fit_additive(np.array(N), np.array(D), np.array(L), 0.86)
print(f"noise {NOISE}: true alpha {alpha} delta {delta}   fitted {fit['alpha']:.3f} {fit['delta']:.3f}")
if NOISE <= 0.01:
    assert abs(fit["alpha"] - alpha) < 0.06 and abs(fit["delta"] - delta) < 0.03
    print("ok")
