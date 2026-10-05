# PS-DTFE / DTFE versus CosmoDTFE.jl

The head-to-head of 2026-09-30 (report: `figures/cosmodtfe_comparison/report.html`).

- `analytic.py`: exact crossed-Zel'dovich-wave fields (point values AND cell averages, mass-weighted velocity).
- `run_cosmo.jl`: CosmoDTFE side (`julia -t 10 --project=<env with CosmoDTFE + HDF5> run_cosmo.jl <dtfe|ps|tune> ...`).
  Tune the BVH depth first (`tune` mode): the default of 9 is ~350 us/point at 7M particles.
- `run_ours.sh`: our side, every run under `/usr/bin/time -l`. `OURS_COMMON` / `PS_EXTRA` for non-periodic regions.
- `run_all.sh <workdir> <repo> <julia env>`: everything, sequentially, on an idle machine. Expects in `<workdir>/data`:
  `zel_{64,128,192}.h5` + `warm.h5` (tests/generate_ps_test_data.py --crossed-waves --amplitude-factor 1.8 --box 100,
  warm = --n 16), `slice2048.bin` / `centres256.bin` (float64 xyz), and for TNG `tng_sub.h5` + its point files.
- `score.py`, `tng.py`, `perf.py`, `tables.py`, `build_report.py`: scores, figures and the report page.
- `rerun_fixed.sh <workdir> <repo>`: re-times only our four stages the 2026-09-30 fixes touched (point values, slice,
  CPU deposit, exact GPU deposit) on all four data sets; copy `out/o*.timing.jsonl` to `out/before_fix/` first, and
  `tables.py` adds the before/after table, `before_after.py <workdir>` prints it.
- `rerun_tng.sh <workdir> <repo>`: re-times our TNG phase-space stages (after the 2026-09-30 region-face and
  alpha-shape fixes, which change the non-periodic region runs).
- Standard-DTFE cell averages (`dtfe_avg_cpu`, `dtfe_avg_gpu`, `dtfe_avg512_cpu`, `dtfe_avg512_gpu` in `run_ours.sh`,
  added 2026-10-01 evening with the rebuilt GPU kernel): the volume-averaged density and velocity at the report grid
  and at 512^3, CPU and GPU; CosmoDTFE has no equivalent. The before/after table takes their "before" (this morning's
  per-tetrahedron kernel) from `out/before_dtfe_<tag>.timing.jsonl`, produced by running `run_ours.sh` with a copy of
  the old binary as `<repo>` and the other stages skipped.
