/* Vendored stand-in for r3d's CMake-generated header (config/r3d-config.h.in).
   R3D_MAX_VERTS bounds the vertex buffer of one clipped polyhedron: a tetrahedron
   clipped by the 6 planes of one grid cell gains at most a few vertices per cut,
   so the upstream default of 512 is far more than the deposit can ever need. */
#ifndef R3D_CONFIG_H_
#define R3D_CONFIG_H_

#define R3D_MAX_VERTS 512

#endif
