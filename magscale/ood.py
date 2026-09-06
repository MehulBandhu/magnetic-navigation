"""Evaluate a trained checkpoint outside its training distribution: other noise levels and
other altitudes, each against the exact floor for that condition.

python -m magscale.ood --run runs/vitxxl_b3.5_h200_D131072_s0_long.json --sigmas 0.3,1,3 --hs 200
python -m magscale.ood --run runs/vitxxl_b3.5_hmix_D131072_s0_alt.json --hs 100,400 --sigmas 1
"""
import argparse
import json
import os

import torch

from .grf import sample_fields, add_noise, make_mask, observe
from .floor import exact_floor
from .models import build
from .train import masked_mse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--sigmas", default="1")
    p.add_argument("--hs", default="200")
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r = json.load(open(a.run))
    c = r["config"]
    model = build(c["model"], c["n"], c["width"], c["depth"], c["heads"], not c["no_altitude"], c["mup"], c["base_width"], c["patch"]).to(device)
    model.load_state_dict(torch.load(a.run.replace(".json", ".pt"), map_location=device))
    model.eval()
    train_hs = [float(x) for x in str(c["h"]).split(",")]
    hmax = max(800.0, max(train_hs))
    scale = c["sigma0"]
    rows = []
    for sigma in [float(s) for s in a.sigmas.split(",")]:
        for h in [float(x) for x in a.hs.split(",")]:
            gen = torch.Generator(device=device).manual_seed(777)
            tot, n = 0.0, 0
            with torch.no_grad():
                for _ in range(a.batches):
                    clean = sample_fields(c["batch"], c["n"], c["beta"], h, c["dx"], c["sigma0"], gen, device)
                    noisy = add_noise(clean, sigma, gen)
                    mask = make_mask(c["mask"], c["batch"], c["n"], c["ratio"], c["spacing"], gen, device)
                    pred = model(observe(noisy, mask, scale), torch.full((c["batch"],), h / hmax, device=device)) * scale
                    tot += masked_mse(pred.float(), clean, mask.float()).sum().item(); n += c["batch"]
            floor = exact_floor(c["n"], c["beta"], h, sigma, c["mask"], c["ratio"], c["spacing"], c["dx"], c["sigma0"], 4)[0]
            rows.append(dict(sigma=sigma, h=h, loss=tot / n, floor=floor, in_training=(h in train_hs and sigma == c["sigma"])))
            print(f"sigma {sigma:4.1f} nT  h {h:5.0f} m  loss {tot / n:8.4f}  floor {floor:8.4f}  ratio {tot / n / floor:6.2f}"
                  f"{'' if rows[-1]['in_training'] else '   (outside the training distribution)'}")
    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(a.run)[:-5]
    json.dump(dict(run=name, train_h=train_hs, train_sigma=c["sigma"], rows=rows), open(f"{a.out}/ood_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
