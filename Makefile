.PHONY: demo test figures fit plots tables p1 floor magnav probes-core probes-widths ood emag2 animations deep_linear compute_optimal sweeps main smalld fixedmask seeds xxs knee extra teacher gap width altitude linear
# from committed results, no weights needed: tests, Problem 1 numerics, floors, fit, plots
test:          ; python -m pytest -q tests
figures:       p1 floor plots deep_linear compute_optimal tables
p1:            ; python -m problem1.checks && python -m problem1.edge_cases
floor:         ; python -m magscale.floor
fit:           ; python -m magscale.fit && python -m magscale.fit --tag _fm
plots:         fit
	 python -m magscale.plots all
tables:        fit compute_optimal deep_linear
	 python scripts/render_tables.py
deep_linear:   ; python -m magscale.deep_linear
animations:    ; python -m magscale.animate all
compute_optimal: ; python -m magscale.compute_optimal
# these add a network column when the release checkpoints are in runs/; without them they write
# *_nonetwork outputs and leave the committed versions alone
demo:          ; python demo.py
magnav:        ; python -m magscale.magnav
# need weights (docs/release.md): the two probed checkpoints with their initial weights, or all 30 width files
probes-core:   ; python -m magscale.probes all --run runs/vitxxl_b3.5_h200_D131072_s0_ref.json --h 200 && python -m magscale.probes all --run runs/vitxxl_b3.5_h200_D131072_s0_long.json --h 200
probes-widths: ; python -m magscale.probes widths
# training
sweeps:        ; python scripts/sweep.py
main:          ; bash scripts/run_list.sh scripts/sweeps/colab_main.txt
smalld:        ; bash scripts/run_list.sh scripts/sweeps/colab_smallD.txt
fixedmask:     ; bash scripts/run_list.sh scripts/sweeps/colab_fixedmask.txt
seeds:         ; bash scripts/run_list.sh scripts/sweeps/colab_seeds.txt
gap:           ; bash scripts/run_list.sh scripts/sweeps/gap.txt
width:         ; bash scripts/run_list.sh scripts/sweeps/width.txt
altitude:      ; bash scripts/run_list.sh scripts/sweeps/altitude.txt
xxs:           ; bash scripts/run_list.sh scripts/sweeps/xxs.txt
knee:          ; bash scripts/run_list.sh scripts/sweeps/knee.txt
extra:         ; bash scripts/run_list.sh scripts/sweeps/extra.txt
teacher:       ; bash scripts/run_list.sh scripts/sweeps/teacher.txt
# evaluation-only checks that need the release checkpoints (and the two altitude-mixture ones)
emag2:         ; python -m magscale.emag2 calibrate && python -m magscale.emag2 synthetic && python -m magscale.emag2 run
ood:           ; python -m magscale.ood --run runs/vitxxl_b3.5_h200_D131072_s0_long.json --sigmas 0.3,1,3 --hs 200 && python -m magscale.ood --run runs/vitxxl_b3.5_hmix_D131072_s0_alt.json --sigmas 1 --hs 100,200,400 && python -m magscale.ood --run runs/vitxxl_b3.5_hmix_D131072_s0_noalt.json --sigmas 1 --hs 100,200,400
linear:        ; bash scripts/run_list.sh scripts/sweeps/linear.txt
