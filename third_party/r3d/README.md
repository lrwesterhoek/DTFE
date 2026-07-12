# r3d (vendored)

`r3d.c` / `r3d.h` from https://github.com/devonmpowell/r3d, commit
`58dfbfb2fd2a89e36e3c7ceb5ca6aef3f9c4c4e6` (2024-09-19), unmodified. Fast, geometrically
robust clipping and analytic volume/moment computation over polyhedra — the exact
conservative voxelization method of **Powell & Abel (2015)**, *An exact general remeshing
scheme applied to physically conservative voxelization*, JCP 297, 340 (arXiv:1412.4941);
report LA-UR-15-26964. Please cite both when using `--ps-exact-deposit` / `--exact-average`
results in research.

Copyright (C) 2015, DOE and Los Alamos National Security, LLC. Permission is granted to the
public to copy and use this software without charge, provided that the Notice and any
statement of authorship in the source headers are reproduced on all copies (see the license
text at the top of `r3d.c` / `r3d.h`).

`r3d-config.h` is not upstream: it stands in for the CMake-generated header and only
defines `R3D_MAX_VERTS` (upstream default 512).

Used by:
- `PS-DTFE --ps-exact-deposit` — exact tetrahedron-cell intersection volumes/moments in the
  phase-space grid deposit (src/CGAL_triangulation/ps_interpolation.cc);
- `DTFE --exact-average` — exact volume-averaged '_a' fields for the standard estimator
  (src/CGAL_triangulation/averaged_interpolation_*.cc).
