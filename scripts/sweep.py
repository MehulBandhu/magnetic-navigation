"""Writes one shell command per line into scripts/sweeps/*.txt. Edit the grids here.
Run a list sequentially with scripts/run_list.sh, or as a SLURM array with scripts/slurm/cuillin_gpu.slurm."""
import os

# width, depth, heads. 0.13M to 38M params for n=64, patch=4
SIZES = {
    "xxs": (32, 2, 2),
    "xs": (64, 2, 2),
    "s": (96, 3, 3),
    "m": (160, 4, 4),
    "l": (256, 6, 4),
    "xl": (384, 8, 6),
    "xxl": (512, 12, 8),
}
SMALL = ["xs", "s", "m", "l"]                    # fit an 11 GB card in fp16
XXS_ONLY = ["xxs"]                               # added later to reach 2.5 orders of magnitude in N
STEPS = {"xxs": 4000, "xs": 4000, "s": 4000, "m": 6000, "l": 8000, "xl": 10000, "xxl": 14000}
LR = {"xxs": 2e-3, "xs": 2e-3, "s": 2e-3, "m": 1e-3, "l": 1e-3, "xl": 6e-4, "xxl": 4e-4}

# the scaling grid: full altitude series at beta=3.5 plus one beta=2.5 condition, three
# dataset sizes, xxl only where it does not memorise. About 11 A100-hours.
CONDITIONS = [(3.5, 0), (3.5, 800), (2.5, 200), (3.5, 200)]
DATASETS = [4096, 32768, 262144]
UNET_WIDTHS = [8, 16, 32, 64]                    # 0.12M to 7.8M
WIDTH_SWEEP = [64, 128, 256, 512, 1024]          # Problem 3C, SP vs muP-inspired at depth 4
ALL_CONDITIONS = [(b, h) for b in [2.5, 3.0, 3.5, 4.0] for h in [0, 100, 200, 400, 800]]

COMMON = "--n 64 --sigma 1.0 --mask random --ratio 0.75 --batch 256 --amp --floor_masks 8"


def name(**kw):
    return "_".join(f"{k}{v}" for k, v in kw.items())


def vit_cmd(size, beta, h, D, seed=0, steps=None, extra="", tag=""):
    w, d, hd = SIZES[size]
    steps = steps or STEPS[size]
    run = name(vit=size, b=beta, h=h, D=D, s=seed) + tag
    return (f"python -m magscale.train --model vit --width {w} --depth {d} --heads {hd} --lr {LR[size]} "
            f"--beta {beta} --h {h} --D {D} --steps {steps} --seed {seed} {COMMON} {extra} --out runs/{run}.json")


def main():
    os.makedirs("sweeps", exist_ok=True)

    grid = [vit_cmd(sz, b, h, D)
            for (b, h) in CONDITIONS for D in DATASETS for sz in SIZES if sz != "xxs" and not (sz == "xxl" and D == DATASETS[0])]
    open("scripts/sweeps/colab_main.txt", "w").write("\n".join(grid) + "\n")

    # below ~1000 fields is where the spectrum-dependent data exponent lives (the linear
    # reference shows delta = 1 above it), so a small-D block for the four small sizes
    small_d = [vit_cmd(sz, b, h, D) for (b, h) in CONDITIONS for D in [256, 1024] for sz in SMALL]
    open("scripts/sweeps/colab_smallD.txt", "w").write("\n".join(small_d) + "\n")

    # the smallest size, added to reach 2.5 orders of magnitude in N (35k to 38.5M)
    xxs = [vit_cmd("xxs", b, h, D) for (b, h) in CONDITIONS for D in [256, 1024] + DATASETS]
    open("scripts/sweeps/xxs.txt", "w").write("\n".join(xxs) + "\n")

    # knee isolation: one factor at a time between m (160w, 4 layers, head dim 40, 6k steps) and
    # l (256w, 6 layers, head dim 64, 8k steps), same condition and data, same rate
    kb, kh, kD = 3.5, 200, 32768
    knee = []
    for tag, w, d, hd, steps in [("m_8ksteps", 160, 4, 4, 8000), ("w256_d4", 256, 4, 4, 6000), ("w160_d6", 160, 6, 4, 6000),
                                 ("l_headdim32", 256, 6, 8, 8000), ("w192_d4_headdim64", 192, 4, 3, 6000)]:
        knee.append(f"python -m magscale.train --model vit --width {w} --depth {d} --heads {hd} --lr 1e-3 --beta {kb} --h {kh} "
                    f"--D {kD} --steps {steps} --seed 0 {COMMON} --out runs/knee_{tag}_b{kb}_h{kh}_D{kD}.json")
    open("scripts/sweeps/knee.txt", "w").write("\n".join(knee) + "\n")

    # two single-run checks: weight decay off (does it drive the NTK scale drop?), noisy targets
    extra = [f"python -m magscale.train --model vit --width 512 --depth 4 --heads 16 --lr 1e-3 --beta 3.5 --h 200 --D 0 --steps 4000 "
             f"{COMMON} --wd 0 --save --out runs/width512_parsp_wd0_b3.5_h200.json",
             vit_cmd("l", 3.5, 200, 32768, extra="--noisy_target", tag="_noisytarget")]
    open("scripts/sweeps/extra.txt", "w").write("\n".join(extra) + "\n")

    # Wiener-teacher runs: targets are the exact conditional mean on fixed masks, so the loss has no
    # floor and its residual after long training bounds approximation error (Problem 2C)
    teach = [vit_cmd("l", 3.5, 200, 0, steps=8000, extra="--teacher 1", tag="_teacher1_8k"),
             vit_cmd("l", 3.5, 200, 0, steps=24000, extra="--teacher 1", tag="_teacher1_24k"),
             vit_cmd("l", 3.5, 200, 0, steps=24000, extra="--teacher 8", tag="_teacher8_24k"),
             vit_cmd("xxl", 3.5, 200, 0, steps=24000, extra="--teacher 1", tag="_teacher1_24k")]
    open("scripts/sweeps/teacher.txt", "w").write("\n".join(teach) + "\n")

    # fixed-mask study: the transformer on exactly the task the linear reference solves
    fm = [vit_cmd(sz, 3.5, 200, D, extra="--fixed_mask", tag="_fm") for D in [256, 1024, 4096, 32768] for sz in SMALL]
    open("scripts/sweeps/colab_fixedmask.txt", "w").write("\n".join(fm) + "\n")

    # seed repeats: run-to-run scatter for the fit uncertainty
    seeds = [vit_cmd(sz, b, h, 32768, seed) for (b, h) in [(3.5, 200), (3.5, 800)] for sz in SMALL for seed in [1, 2]]
    seeds += [vit_cmd("xl", 3.5, 200, 32768, seed) for seed in [1, 2]]
    open("scripts/sweeps/colab_seeds.txt", "w").write("\n".join(seeds) + "\n")

    # Problem 2C: move one thing at a time on the largest model, one condition
    b, h, D = 3.5, 200, 131072
    gap = [
        vit_cmd("xxl", b, h, D, steps=24000, tag="_long", extra="--save"),                 # optimisation error
        vit_cmd("xxl", b, h, 1048576, steps=8000, tag="_bigdata", extra="--save"),        # data
        vit_cmd("xxl", b, h, D, steps=8000, extra="--save", tag="_ref"),                   # reference with weights
        vit_cmd("xxl", b, h, D, steps=8000, extra="--save --no_altitude", tag="_noalt"),   # conditioning ablation
        vit_cmd("xxl", b, h, D, steps=8000, extra="--save --mask lines --spacing 4", tag="_lines"),
    ]
    unet = [f"python -m magscale.train --model unet --width {w} --depth 3 --lr 1e-3 --beta {b} --h {h} --D {D} "
            f"--steps 6000 {COMMON} --save --out runs/{name(unet=w, b=b, h=h, D=D, s=0)}.json"
            for w in UNET_WIDTHS]                                                           # approximation error
    open("scripts/sweeps/gap.txt", "w").write("\n".join(gap + unet) + "\n")

    # altitude conditioning ablation on a three-altitude mixture
    alt = [f"python -m magscale.train --model vit --width 512 --depth 12 --heads 8 --lr 4e-4 --beta 3.5 --h 0,200,800 "
           f"--D 131072 --steps 8000 --seed 0 {COMMON} --save {flag} --out runs/vitxxl_b3.5_hmix_D131072_s0_{tag}.json"
           for flag, tag in [("", "alt"), ("--no_altitude", "noalt")]]
    open("scripts/sweeps/altitude.txt", "w").write("\n".join(alt) + "\n")

    # Problem 3C: kernel regime or not. Same rate for all widths under SP; muP-inspired at two base rates.
    width = []
    for w in WIDTH_SWEEP:
        for par, lr in [("sp", "1e-3"), ("mup", "1e-3"), ("mup4e3", "4e-3")]:
            mup = "--mup --base_width 64" if par.startswith("mup") else ""
            width.append(f"python -m magscale.train --model vit --width {w} --depth 4 --heads {max(2, w // 32)} "
                         f"--lr {lr} --beta {b} --h {h} --D 0 --steps 4000 {COMMON} {mup} --save "
                         f"--out runs/{name(width=w, par=par, b=b, h=h)}.json")
    open("scripts/sweeps/width.txt", "w").write("\n".join(width) + "\n")

    # CPU: the linear learner's learning curve on every condition (Problem 3B reference)
    lin = [f"python -m magscale.linear_ref --beta {b} --h {h} --seed {seed} --out results/linear_b{b}_h{h}_s{seed}.json"
           for (b, h) in ALL_CONDITIONS for seed in range(2)]
    open("scripts/sweeps/linear.txt", "w").write("\n".join(lin) + "\n")

    for f in ["colab_main", "colab_smallD", "colab_fixedmask", "colab_seeds", "xxs", "knee", "extra", "teacher", "gap", "width", "altitude", "linear"]:
        print(f, sum(1 for _ in open(f"scripts/sweeps/{f}.txt")), "runs")


if __name__ == "__main__":
    main()
