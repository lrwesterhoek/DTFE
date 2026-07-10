/* Phase-space DTFE grid interpolation. Triangulation is built in Lagrangian space;
   Eulerian simplices can overlap (multi-stream), so we iterate over all cells and
   accumulate each simplex's contribution to every grid point it contains. */

#ifdef PHASE_SPACE

// PS-DTFE is validated only in 3D; the NO_DIM==2 path compiles but is untested.
#if NO_DIM==2
#warning "PS-DTFE (PHASE_SPACE) is not validated for NO_DIM==2; the 2D path is untested."
#endif

#include "triangulation_common.h"

#include <cstring>   // memcpy in the in-place cost-sort permutation

#if defined(PS_GPU) && NO_DIM==3
#include "gpu_host.h"   // GPU deposit (METAL=1/CUDA=1/HIP=1 builds); CPU deposit remains the fallback
#endif


/* Interpolates fields onto a regular grid via PS-DTFE: scatters each finite cell's
   contribution to overlapping grid points so multi-stream regions are handled.
   nSub = sub-samples per axis for volume-averaged fields (1 = unaveraged).
   mayClearDT = this is the last use of an internally-owned dt: clear it right after the
   deposit's last read (GPU-success or CPU-loop end) to free ~650 B/vertex before the
   stats/normalization here and the caller's merge into the shared grids. */
void interpolateGrid_phaseSpace(DT &dt,
                                User_options &userOptions,
                                Quantities *quantities,
                                Field &field,
                                int nSub,
                                bool mayClearDT)
{
    MESSAGE::Message message( userOptions.verboseLevel );
    size_t const *nGrid = &(userOptions.gridSize[0]);
    Box boxCoordinates = userOptions.region;

    // Averaged fields sample each grid cell on an nSub^NO_DIM sub-grid (nSub==1 = cell centre only).
    if ( nSub < 1 ) nSub = 1;
    size_t nSamplesPerCell = 1;
    for (int d = 0; d < NO_DIM; ++d) nSamplesPerCell *= size_t(nSub);

    Real dx[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d)
        dx[d] = (boxCoordinates[2*d+1] - boxCoordinates[2*d]) / nGrid[d];
    Real cellVolume = Real(1.);
    for (int d = 0; d < NO_DIM; ++d) cellVolume *= dx[d];

    size_t totalGrid = 1;
    for (int d = 0; d < NO_DIM; ++d) totalGrid *= nGrid[d];

    // Every pass below walks the finite cells DIRECTLY via the CGAL iterator (all consuming
    // loops are serial, and iteration order is deterministic for a fixed triangulation), so
    // no cellHandles vector is materialized -- that was 8 B x all finite cells (~0.4 GB per
    // concurrent partition at production scale) held for the whole interpolation.
    size_t noTotalCells = 0;
#if NO_DIM==2
    noTotalCells = dt.number_of_faces();
#elif NO_DIM==3
    noTotalCells = dt.number_of_finite_cells();
#endif
#if NO_DIM==2
    #define PS_FOREACH_FINITE_CELL for (DT::Finite_faces_iterator itC = dt.finite_faces_begin(); itC != dt.finite_faces_end(); ++itC)
#elif NO_DIM==3
    #define PS_FOREACH_FINITE_CELL for (DT::Finite_cells_iterator itC = dt.finite_cells_begin(); itC != dt.finite_cells_end(); ++itC)
#endif

    // psUseSubgrid: store only the Eulerian bbox of cells this partition touches, to bound peak
    // memory. Marks per axis which grid planes any kept cell overlaps (same filter as the scatter,
    // so the box covers every written cell); an axis reaching both ends stays full. off -> full grid.
    size_t subOrigin[NO_DIM], subDims[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d) { subOrigin[d] = 0; subDims[d] = nGrid[d]; }
    // cells surviving the dummy/ownership/degeneracy filters, counted by the subgrid pass below;
    // used to right-size the Metal tet arrays (reserving for ALL cells over-allocates ~4x, since
    // most padded-partition cells are filtered out). Upper bound when the pass does not run.
    size_t psKeptCells = noTotalCells;
    if ( userOptions.psUseSubgrid )
    {
        psKeptCells = 0;
        std::vector<char> touched[NO_DIM];
        for (int d = 0; d < NO_DIM; ++d) touched[d].assign(nGrid[d], char(0));
        PS_FOREACH_FINITE_CELL
        {
            Cell_handle cell = itC;
#ifdef TEST_PADDING
            { bool dummy=false; for (int v=0;v<=NO_DIM;++v) if (cell->vertex(v)->info().isDummy()){dummy=true;break;} if (dummy) continue; }
#endif
            if ( !userOptions.lagrangianRegion.isNullBox() )
            {
                double cen[NO_DIM]; for (int d=0;d<NO_DIM;++d) cen[d]=0.;
                for (int v=0;v<=NO_DIM;++v) for (int d=0;d<NO_DIM;++d) cen[d]+=double(cell->vertex(v)->point()[d]);
                bool owned=true;
                for (int d=0;d<NO_DIM;++d){ cen[d]/=double(NO_DIM+1); if (cen[d]<userOptions.lagrangianRegion[2*d]||cen[d]>=userOptions.lagrangianRegion[2*d+1]){owned=false;break;} }
                if (!owned) continue;
            }
            { bool bad=false; for (int v=0;v<=NO_DIM;++v) if (cell->vertex(v)->info().density()<=Real(0.)){bad=true;break;} if (bad && userOptions.periodic) continue; }   // non-periodic keeps hull cells (volume-ratio density)
            Real ep[NO_DIM+1][NO_DIM];
            for (int v=0;v<=NO_DIM;++v) for (int d=0;d<NO_DIM;++d) ep[v][d]=cell->vertex(v)->info().eulerianPosition(d);
            if ( userOptions.periodic )
            {
                Real boxLen[NO_DIM]; for (int d=0;d<NO_DIM;++d) boxLen[d]=boxCoordinates[2*d+1]-boxCoordinates[2*d];
                for (int v=1;v<=NO_DIM;++v) for (int d=0;d<NO_DIM;++d){ Real diff=ep[v][d]-ep[0][d]; if(diff>boxLen[d]*Real(0.5))ep[v][d]-=boxLen[d]; if(diff<-boxLen[d]*Real(0.5))ep[v][d]+=boxLen[d]; }
            }
            double Ax2[NO_DIM][NO_DIM];
            for (int v=0;v<NO_DIM;++v) for (int i=0;i<NO_DIM;++i) Ax2[v][i]=double(ep[v+1][i])-double(ep[0][i]);
            { double avgEdge2=0.; for(int v=0;v<NO_DIM;++v){double l2=0.;for(int i=0;i<NO_DIM;++i)l2+=Ax2[v][i]*Ax2[v][i];avgEdge2+=l2;} avgEdge2/=NO_DIM; double edgeScale=avgEdge2*std::sqrt(avgEdge2); if (std::fabs(determinant(Ax2)) < 1.e-6*edgeScale) continue; }
            ++psKeptCells;
            Real eLo[NO_DIM],eHi[NO_DIM];
            for (int d=0;d<NO_DIM;++d){eLo[d]=ep[0][d];eHi[d]=ep[0][d];}
            for (int v=1;v<=NO_DIM;++v) for (int d=0;d<NO_DIM;++d){ if(ep[v][d]<eLo[d])eLo[d]=ep[v][d]; if(ep[v][d]>eHi[d])eHi[d]=ep[v][d]; }
            for (int d=0;d<NO_DIM;++d)
            {
                int iLo=int(floor((eLo[d]-boxCoordinates[2*d])/dx[d]));
                int iHi=int(floor((eHi[d]-boxCoordinates[2*d])/dx[d]))+1;
                if (nSub>1){iLo-=1;iHi+=1;}
                if (!userOptions.periodic){ if(iLo<0)iLo=0; if(iHi>(int)nGrid[d])iHi=(int)nGrid[d]; }
                for (int g=iLo; g<iHi; ++g){ int w = userOptions.periodic ? ((g%(int)nGrid[d]+(int)nGrid[d])%(int)nGrid[d]) : g; if (w>=0 && w<(int)nGrid[d]) touched[d][w]=char(1); }
            }
        }
        for (int d = 0; d < NO_DIM; ++d)
        {
            int lo=-1, hi=-1;
            for (int g=0; g<(int)nGrid[d]; ++g) if (touched[d][g]){ if(lo<0)lo=g; hi=g; }
            if (lo<0) { subOrigin[d]=0; subDims[d]=nGrid[d]; }                                 // no cells touched: keep full axis (harmless)
            else if (touched[d][0] && touched[d][nGrid[d]-1]) { subOrigin[d]=0; subDims[d]=nGrid[d]; }  // spans both ends (likely wraps): keep full axis
            else { subOrigin[d]=(size_t)lo; subDims[d]=(size_t)(hi-lo+1); }                    // crop to the touched [lo,hi] span
        }
    }
    size_t subTotal = 1; for (int d = 0; d < NO_DIM; ++d) subTotal *= subDims[d];
    totalGrid = subTotal;   // all allocations and loops below use the sub-grid size

    // Allocate the requested output fields, zeroed since the cell loop accumulates into them.
    if ( field.density )
        quantities->density.assign(totalGrid, Real(0.));
    // Dispersion needs <v>, so allocate velocity storage even when only dispersion was requested.
    bool const haveVel = field.velocity || field.velocity_dispersion;
    if ( haveVel )
        quantities->velocity.assign(totalGrid, Pvector<Real,noVelComp>::zero());
    if ( field.velocity_gradient )
        quantities->velocity_gradient.assign(totalGrid, Pvector<Real,noGradComp>::zero());
    if ( field.velocity_dispersion )
        quantities->velocity_dispersion.assign(totalGrid, Pvector<Real,noDispComp>::zero());
#ifdef SCALAR
    if ( field.scalar )
        quantities->scalar.assign(totalGrid, Pvector<Real,noScalarComp>::zero());
    if ( field.scalar_gradient )
        quantities->scalar_gradient.assign(totalGrid, Pvector<Real,noScalarGradComp>::zero());
#endif

    // Per-grid-point count of overlapping streams (diagnostic; reported as the stream_count field).
    std::vector<int> streamCount(totalGrid, 0);

    // Count of cells dropped because their Eulerian simplex is non-invertible (see the check below).
    size_t nDegenerateInverse = 0;

    // multi-stream velocity/scalar = mass-weighted mean sum(rho_s f_s)/sum(rho_s): accumulate
    // density-weighted moments + mass weight sum(rho_s). Serial path normalizes at function end;
    // psDeferNormalization leaves moments un-normalized so they sum linearly across partitions
    // (the mean is non-linear), normalizing once later.
    bool const deferNorm = userOptions.psDeferNormalization;
    bool const needWeight = field.velocity || field.velocity_gradient || field.velocity_dispersion
#ifdef SCALAR
                            || field.scalar || field.scalar_gradient
#endif
                            ;
    // when the density field is selected, its accumulator receives exactly the same per-cell
    // mass sums as massWeight would (both add massShare at the same sites), so the weight
    // ALIASES the density grid instead of allocating a second full grid (4 B/cell). The
    // deferNorm caller reconstructs the weight from the summed density (normalizePhaseSpace
    // scale argument), since density is only converted to rho/rho_bar after weighting here.
    bool const weightIsDensity = needWeight && field.density;
    std::vector<Real> massWeight;
    if ( needWeight && !weightIsDensity ) massWeight.assign(totalGrid, Real(0.));

    // Reusable per-tetrahedron scratch for the mass-conserving deposit (see the cell loop): the flat
    // grid index and Eulerian offset (rel = samplePos - vertex0) of every in-simplex sample point.
    std::vector<size_t> insideFlat;
    std::vector<Real>   insideRel;   // insideFlat.size() * NO_DIM, row-major per sample
    // --ps-linear-deposit: per-sample weight = the DTFE-interpolated linear density at the sample
    // (clamped >= 0); PASS 2 renormalizes the weights to the tet mass, so the deposited total
    // still equals tetMass exactly and mass conservation is untouched.
    bool const psLinear = userOptions.psLinearDeposit;
    std::vector<Real>   insideW;

    if ( not userOptions.psSuppressGridStats )
        message << "\n" << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset()
                << " Interpolating fields to grid by iterating over all Delaunay cells ...\n" << MESSAGE::Flush;

    // GPU deposit (--ps-metal / --ps-gpu, GPU builds): extract flat per-tet arrays with EXACTLY the CPU
    // loop's filters, dispatch metal/ps_deposit.metal::depositFields (validated to mirror the CPU
    // scatter incl. the partition sub-grid), and copy the moment grids back. Any failure (no
    // device, kernel compile, buffer alloc) falls back to the CPU loop below with a warning.
    bool metalDeposited = false;
#if defined(PS_GPU) && NO_DIM==3
    bool tryMetal = userOptions.psUseMetal;
#ifdef SCALAR
    if ( tryMetal && (field.scalar || field.scalar_gradient) )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "--ps-metal does not support the scalar fields; using the CPU deposit for this pass.\n" << MESSAGE::EndWarning;
        tryMetal = false;
    }
#endif
    if ( tryMetal )
    {
        // which moment grids this run needs: unselected ones are neither extracted, allocated
        // (host or GPU; m2 is 24 B/cell, grad 36 B/cell) nor copied back
        bool const fVel  = haveVel;
        bool const fDisp = field.velocity_dispersion;
        bool const fGrad = field.velocity_gradient;
        bool const needsVelArrays = fVel || fDisp || fGrad;

        // reserve for the cells that survive the filters (counted by the subgrid pass), not all
        // finite cells -- the padded-partition majority is filtered out (~4x over-allocation)
        std::vector<float> tetVerts, tetVels, tetMasses, tetDens;
        tetVerts.reserve( psKeptCells*12 );
        if ( needsVelArrays )
            tetVels.reserve( psKeptCells*12 );
        tetMasses.reserve( psKeptCells );
        if ( psLinear )
            tetDens.reserve( psKeptCells*4 );
        PS_FOREACH_FINITE_CELL
        {
            Cell_handle cell = itC;
#ifdef TEST_PADDING
            { bool dummy=false; for (int v=0;v<=NO_DIM;++v) if (cell->vertex(v)->info().isDummy()){dummy=true;break;} if (dummy) continue; }
#endif
            if ( !userOptions.lagrangianRegion.isNullBox() )
            {
                double cen[NO_DIM]; for (int d=0;d<NO_DIM;++d) cen[d]=0.;
                for (int v=0;v<=NO_DIM;++v) for (int d=0;d<NO_DIM;++d) cen[d]+=double(cell->vertex(v)->point()[d]);
                bool owned=true;
                for (int d=0;d<NO_DIM;++d){ cen[d]/=double(NO_DIM+1); if (cen[d]<userOptions.lagrangianRegion[2*d]||cen[d]>=userOptions.lagrangianRegion[2*d+1]){owned=false;break;} }
                if (!owned) continue;
            }
            bool hadBadVertex = false;
            { bool bad=false; for (int v=0;v<=NO_DIM;++v) if (cell->vertex(v)->info().density()<=Real(0.)){bad=true;break;} if (bad && userOptions.periodic) continue; hadBadVertex = bad; }
            Real ep[NO_DIM+1][NO_DIM];
            for (int v=0;v<=NO_DIM;++v) for (int d=0;d<NO_DIM;++d) ep[v][d]=cell->vertex(v)->info().eulerianPosition(d);
            if ( userOptions.periodic )
            {
                Real boxLen[NO_DIM]; for (int d=0;d<NO_DIM;++d) boxLen[d]=boxCoordinates[2*d+1]-boxCoordinates[2*d];
                for (int v=1;v<=NO_DIM;++v) for (int d=0;d<NO_DIM;++d){ Real diff=ep[v][d]-ep[0][d]; if(diff>boxLen[d]*Real(0.5))ep[v][d]-=boxLen[d]; if(diff<-boxLen[d]*Real(0.5))ep[v][d]+=boxLen[d]; }
            }
            double Ax[NO_DIM][NO_DIM];
            for (int v=0;v<NO_DIM;++v) for (int i=0;i<NO_DIM;++i) Ax[v][i]=double(ep[v+1][i])-double(ep[0][i]);
            { double avgEdge2=0.; for(int v=0;v<NO_DIM;++v){double l2=0.;for(int i=0;i<NO_DIM;++i)l2+=Ax[v][i]*Ax[v][i];avgEdge2+=l2;} avgEdge2/=NO_DIM; double edgeScale=avgEdge2*std::sqrt(avgEdge2); if (std::fabs(determinant(Ax)) < 1.e-6*edgeScale) continue; }
            {
                Real inv[NO_DIM][NO_DIM]; matrixInverse(Ax,inv);
                bool zero=true; for (int a=0;a<NO_DIM&&zero;++a) for (int b=0;b<NO_DIM;++b) if (inv[a][b]!=Real(0.)){zero=false;break;}
                if (zero) { ++nDegenerateInverse; continue; }
            }
            double Lag[NO_DIM][NO_DIM];
            for (int v=0;v<NO_DIM;++v) for (int i=0;i<NO_DIM;++i) Lag[v][i]=double(cell->vertex(v+1)->point()[i])-double(cell->vertex(0)->point()[i]);
            float tm = float( double(userOptions.averageDensity)*std::fabs(determinant(Lag))/factorial(NO_DIM) );
            if (tm<=0.f) continue;
            for (int v=0;v<=NO_DIM;++v) for (int d=0;d<NO_DIM;++d) tetVerts.push_back(float(ep[v][d]));
            if (needsVelArrays)
                for (int v=0;v<=NO_DIM;++v) for (size_t j=0;j<noVelComp;++j) tetVels.push_back(float(cell->vertex(v)->info().velocity(j)));
            if (psLinear)
            {
                // hull cells (non-periodic volume-ratio density) have a constant profile:
                // pass equal weights, which reduces to the uniform deposit for that tet
                if (hadBadVertex)
                    for (int v=0;v<=NO_DIM;++v) tetDens.push_back(1.f);
                else
                    for (int v=0;v<=NO_DIM;++v) tetDens.push_back(float(cell->vertex(v)->info().density()));
            }
            tetMasses.push_back(tm);
        }

        // Sort tetrahedra by grid-footprint cost before dispatch. A GPU chunk finishes when its
        // SLOWEST thread finishes: one stretched void tet (spanning thousands of grid cells) mixed
        // among cheap halo tets leaves most of the GPU idle until the chunk boundary -- measured as
        // a ~5x underutilization on real data with watchdog-safe short chunks. Cost-ordering makes
        // every chunk's threads uniformly sized, so all GPU cores stay busy, and the adaptive
        // chunk-size controller tracks a smooth monotone cost curve instead of a noisy mix.
        {
            size_t const nT = tetMasses.size();
            std::vector< std::pair<float,uint32_t> > order(nT);
            for (size_t t = 0; t < nT; ++t)
            {
                float lo[3], hi[3];
                for (int d = 0; d < 3; ++d) { lo[d] = tetVerts[t*12+d]; hi[d] = lo[d]; }
                for (int v = 1; v < 4; ++v)
                    for (int d = 0; d < 3; ++d)
                    {
                        float const cc = tetVerts[t*12+v*3+d];
                        if (cc < lo[d]) lo[d] = cc;
                        if (cc > hi[d]) hi[d] = cc;
                    }
                float cells = 1.f;
                for (int d = 0; d < 3; ++d)
                    cells *= (hi[d]-lo[d]) / float(dx[d]) + (nSub > 1 ? 3.f : 1.f);
                order[t] = std::make_pair(cells, uint32_t(t));
            }
            std::sort(order.begin(), order.end());
            // apply the permutation IN PLACE by cycle-walking all arrays in lockstep: a gather
            // into a fresh array would transiently hold a second nT*12-float copy (~90 MB per
            // million tets); this needs only one element of scratch plus an nT-byte mask
            {
                std::vector<char> visited(nT, 0);
                bool const haveU = not tetVels.empty();
                bool const haveD = not tetDens.empty();
                float tmpV[12], tmpU[12], tmpD[4];
                for (size_t start = 0; start < nT; ++start)
                {
                    if (visited[start]) continue;
                    std::memcpy(tmpV, &tetVerts[start*12], 12*sizeof(float));
                    if (haveU) std::memcpy(tmpU, &tetVels[start*12], 12*sizeof(float));
                    if (haveD) std::memcpy(tmpD, &tetDens[start*4], 4*sizeof(float));
                    float const tmpM = tetMasses[start];
                    size_t i = start;
                    while (true)
                    {
                        size_t const src = order[i].second;
                        visited[i] = 1;
                        if (src == start)
                        {
                            std::memcpy(&tetVerts[i*12], tmpV, 12*sizeof(float));
                            if (haveU) std::memcpy(&tetVels[i*12], tmpU, 12*sizeof(float));
                            if (haveD) std::memcpy(&tetDens[i*4], tmpD, 4*sizeof(float));
                            tetMasses[i] = tmpM;
                            break;
                        }
                        std::memcpy(&tetVerts[i*12], &tetVerts[src*12], 12*sizeof(float));
                        if (haveU) std::memcpy(&tetVels[i*12], &tetVels[src*12], 12*sizeof(float));
                        if (haveD) std::memcpy(&tetDens[i*4], &tetDens[src*4], 4*sizeof(float));
                        tetMasses[i] = tetMasses[src];
                        i = src;
                    }
                }
            }
        }

        double bl[3]  = { double(boxCoordinates[0]), double(boxCoordinates[2]), double(boxCoordinates[4]) };
        double dxv[3] = { double(dx[0]), double(dx[1]), double(dx[2]) };
        size_t const nTetsDeposited = tetMasses.size();   // psGpuDepositFields consumes (frees) the arrays
        PSGpuGrids gpuOut;
        std::string metalErr;
        if ( psGpuDepositFields(tetVerts, tetVels, tetMasses, tetDens, bl, dxv,
                                  nGrid, subOrigin, subDims, nSub,
                                  userOptions.periodic, fVel, fDisp, fGrad, psLinear, gpuOut, metalErr) )
        {
            if ( not userOptions.psSuppressGridStats )
                message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " " << gpuBackendName() << " GPU deposit ("
                        << MESSAGE::cMagenta() << gpuDeviceName() << MESSAGE::cReset() << "), "
                        << MESSAGE::cMagenta() << nTetsDeposited << MESSAGE::cReset() << " tetrahedra.\n" << MESSAGE::Flush;
            for (size_t i = 0; i < totalGrid; ++i)
            {
                if (field.density) quantities->density[i] = Real(gpuOut.mass[i]);
                if (needWeight && !weightIsDensity) massWeight[i] = Real(gpuOut.mass[i]);
                if (haveVel)
                    for (size_t j = 0; j < noVelComp; ++j)
                        quantities->velocity[i][j] = Real(gpuOut.mom[i*3+j]);
                if (field.velocity_gradient)
                    for (size_t q = 0; q < noGradComp; ++q)
                        quantities->velocity_gradient[i][q] = Real(gpuOut.grad[i*9+q]);
                if (field.velocity_dispersion)
                    for (size_t cc = 0; cc < noDispComp; ++cc)
                        quantities->velocity_dispersion[i][cc] = Real(gpuOut.m2[i*6+cc]);
                streamCount[i] = int(gpuOut.streams[i]);
            }
            metalDeposited = true;
        }
        else
        {
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "PS-DTFE GPU deposit unavailable (" << metalErr << "); using the CPU deposit.\n" << MESSAGE::EndWarning;
        }
    }
#else
    if ( userOptions.psUseMetal )
    {
        static bool warned = false;
        if ( !warned )
        {
            warned = true;
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "--ps-gpu/--ps-metal requested but this binary was built without GPU support (rebuild with METAL=1, CUDA=1 or HIP=1; or NO_DIM!=3); using the CPU deposit.\n" << MESSAGE::EndWarning;
        }
    }
#endif // PS_GPU

    // per-cell scatter is serial; parallelism is one level up over Lagrangian partitions
    // (DTFE.cpp). A per-cell OpenMP scatter was memory-bandwidth bound and gave no speedup.
    if ( !metalDeposited )
    PS_FOREACH_FINITE_CELL
    {
        Cell_handle cell = itC;

#ifdef TEST_PADDING
        // Skip cells touching a dummy padding vertex.
        bool hasDummy = false;
        for (int v = 0; v <= NO_DIM; ++v)
            if (cell->vertex(v)->info().isDummy()) { hasDummy = true; break; }
        if (hasDummy) { continue; }
#endif

        // Lagrangian-partition ownership: padding zones overlap, so keep a cell only if its
        // Lagrangian centroid (not an arbitrary vertex) lies in the primary box -> tiles without double-counting
        if ( !userOptions.lagrangianRegion.isNullBox() )
        {
            double cen[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d) cen[d] = 0.;
            for (int v = 0; v <= NO_DIM; ++v)
                for (int d = 0; d < NO_DIM; ++d)
                    cen[d] += double(cell->vertex(v)->point()[d]);
            bool owned = true;
            for (int d = 0; d < NO_DIM; ++d)
            {
                cen[d] /= double(NO_DIM + 1);
                if ( cen[d] < userOptions.lagrangianRegion[2*d] || cen[d] >= userOptions.lagrangianRegion[2*d+1] )
                { owned = false; break; }
            }
            if (!owned) { continue; }
        }

        // zero/negative-density vertex sits on the Lagrangian convex hull. Periodic: artefact, drop the
        // cell. Non-periodic: real cloud surface, give it a constant volume-ratio density (avgDensity * V_Lag/V_Eul).
        bool useVolumeRatioDensity = false;
        {
            bool hasBadVertex = false;
            for (int v = 0; v <= NO_DIM; ++v)
                if (cell->vertex(v)->info().density() <= Real(0.)) { hasBadVertex = true; break; }
            if (hasBadVertex)
            {
                if ( userOptions.periodic ) { continue; }
                useVolumeRatioDensity = true;
            }
        }

        // Gather the Eulerian vertex positions; the minimum-image convention (below) keeps boundary
        // cells from spuriously spanning the whole periodic box.
        Real eulerPos[NO_DIM+1][NO_DIM];
        for (int v = 0; v <= NO_DIM; ++v)
            for (int d = 0; d < NO_DIM; ++d)
                eulerPos[v][d] = cell->vertex(v)->info().eulerianPosition(d);

        if ( userOptions.periodic )
        {
            Real boxLen[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d)
                boxLen[d] = boxCoordinates[2*d+1] - boxCoordinates[2*d];

            // Wrap vertices 1..NO_DIM to their nearest image of vertex 0.
            for (int v = 1; v <= NO_DIM; ++v)
                for (int d = 0; d < NO_DIM; ++d)
                {
                    Real diff = eulerPos[v][d] - eulerPos[0][d];
                    if (diff >  boxLen[d] * Real(0.5)) eulerPos[v][d] -= boxLen[d];
                    if (diff < -boxLen[d] * Real(0.5)) eulerPos[v][d] += boxLen[d];
                }
        }

        // Eulerian edge matrix (rows = vertices 1..NO_DIM relative to vertex 0, possibly wrapped).
        double Ax[NO_DIM][NO_DIM];
        for (int v = 0; v < NO_DIM; ++v)
            for (int i = 0; i < NO_DIM; ++i)
                Ax[v][i] = double(eulerPos[v+1][i]) - double(eulerPos[0][i]);

        double cellDet = determinant(Ax);
        double cellAbsDet = std::fabs(cellDet);

        // drop only degenerate cells (near-zero Eulerian volume -> divergent 1/volume density).
        // |det| < tol*(mean edge)^3 is resolution-independent; caustics stay well above it.
        double const DEGENERATE_DET_TOL = 1.e-6;
        {
            double avgEdge2 = 0.;
            for (int v = 0; v < NO_DIM; ++v)
            {
                double len2 = 0.;
                for (int i = 0; i < NO_DIM; ++i)
                    len2 += Ax[v][i] * Ax[v][i];
                avgEdge2 += len2;
            }
            avgEdge2 /= NO_DIM;
            double edgeScale = avgEdge2 * std::sqrt(avgEdge2); // avgEdge^3
            if (cellAbsDet < DEGENERATE_DET_TOL * edgeScale) { continue; }
        }

        Real posMatInv[NO_DIM][NO_DIM];
        matrixInverse(Ax, posMatInv);

        // matrixInverse() returns the ALL-ZERO matrix for a (near-)singular Eulerian simplex (|det|
        // below REAL_PRECISSION). A zero inverse makes every barycentric coordinate below exactly 0,
        // which the point-in-simplex test then reads as "inside" for EVERY grid point in the cell's
        // bounding box -- so the cell floods its whole axis-aligned bbox. That is the source of the
        // grid-aligned "square" artefacts: worst at caustics (flat, collapsed cells cluster there) and
        // amplified by sub-sampling (nSub>1 puts more sample points inside the bbox, so the averaged
        // '_a' fields look even worse and their mean density is inflated). A collapsed cell has no
        // reliable Eulerian volume, so drop it; an adjacent non-degenerate stream covers the region.
        // NOTE: this must stay consistent with matrixInverse()'s degeneracy criterion -- the earlier
        // relative determinant pre-filter alone does NOT catch every cell that matrixInverse() zeroes.
        {
            bool singularInverse = true;
            for (int a = 0; a < NO_DIM && singularInverse; ++a)
                for (int b = 0; b < NO_DIM; ++b)
                    if (posMatInv[a][b] != Real(0.)) { singularInverse = false; break; }
            if (singularInverse) { ++nDegenerateInverse; continue; }
        }

        // PS-DTFE per-tetrahedron MASS (Abel/Hahn/Shandarin tetrahedra method): each Lagrangian flow
        // element carries a constant mass m = averageDensity * V_lag, conserved as it maps to its
        // present-day Eulerian simplex (Eulerian volume V_eul). We deposit this finite MASS onto the
        // grid (mass-conserving; see the scatter below), instead of painting the per-tetrahedron
        // density rho = m / V_eul at grid points: at a caustic V_eul -> 0 so rho -> ~1e7, and
        // point-sampling that density either misses the thin simplex (mass undercounted) or splats
        // ~1e7 over a whole grid cell (the grid-aligned "square" over-densities, worse under
        // sub-sampling because more sample points fall in the cell). Depositing m is exact at any nSub.
        Real tetMass;
        {
            double Lag[NO_DIM][NO_DIM];
            for (int v = 0; v < NO_DIM; ++v)
                for (int i = 0; i < NO_DIM; ++i)
                    Lag[v][i] = double(cell->vertex(v+1)->point()[i]) - double(cell->vertex(0)->point()[i]);
            double absDetLag = std::fabs(determinant(Lag));
            tetMass = Real( double(userOptions.averageDensity) * absDetLag / factorial(NO_DIM) );  // = rho_bar * V_lag
            if (tetMass < Real(0.)) tetMass = Real(0.);
        }
        (void)useVolumeRatioDensity;   // retained above only to drop/flag hull cells

        // Eulerian bounding box of this cell (from the wrapped positions).
        Real eMin[NO_DIM], eMax[NO_DIM];
        for (int d = 0; d < NO_DIM; ++d)
        {
            eMin[d] = eulerPos[0][d];
            eMax[d] = eulerPos[0][d];
        }
        for (int v = 1; v <= NO_DIM; ++v)
            for (int d = 0; d < NO_DIM; ++d)
            {
                if (eulerPos[v][d] < eMin[d]) eMin[d] = eulerPos[v][d];
                if (eulerPos[v][d] > eMax[d]) eMax[d] = eulerPos[v][d];
            }

        // Grid index range overlapping the bounding box (periodic indices are wrapped below).
        int iMin[NO_DIM], iMax[NO_DIM];
        bool outsideGrid = false;
        for (int d = 0; d < NO_DIM; ++d)
        {
            iMin[d] = int(floor((eMin[d] - boxCoordinates[2*d]) / dx[d]));
            iMax[d] = int(floor((eMax[d] - boxCoordinates[2*d]) / dx[d])) + 1;
            if (nSub > 1) { iMin[d] -= 1; iMax[d] += 1; }   // sub-sample points can lie in edge cells
            if (!userOptions.periodic)
            {
                if (iMin[d] < 0) iMin[d] = 0;
                if (iMax[d] > (int)nGrid[d]) iMax[d] = (int)nGrid[d];
                if (iMin[d] >= (int)nGrid[d] || iMax[d] <= 0) { outsideGrid = true; break; }
            }
        }
        if (outsideGrid) { continue; }

        // (density is deposited as the per-tetrahedron mass tetMass computed above; no per-vertex
        //  density gradient is needed for the scatter.)

        // Constant velocity gradient (also needed to evaluate the per-stream velocity for the dispersion).
        Real velGrad[NO_DIM][noVelComp];
        if (field.velocity || field.velocity_gradient || field.velocity_dispersion)
        {
            Vertex_handle base = cell->vertex(0);
            Real temp[NO_DIM][noVelComp];
            for (int v = 0; v < NO_DIM; ++v)
                for (size_t i = 0; i < noVelComp; ++i)
                    temp[v][i] = cell->vertex(v+1)->info().velocity(i) - base->info().velocity(i);
            matrixMultiplication<noVelComp>(posMatInv, temp, velGrad);
        }

        // --ps-linear-deposit: constant gradient of the DTFE vertex densities across this cell
        // (same affine convention as the velocity above). Hull cells (non-periodic volume-ratio
        // density) have a constant profile, which degrades to the uniform deposit below.
        Real denBase = Real(0.), denGrad[NO_DIM] = { Real(0.) };
        bool linearProfile = false;
        if ( psLinear && !useVolumeRatioDensity )
        {
            denBase = cell->vertex(0)->info().density();
            Real temp[NO_DIM];
            for (int v = 0; v < NO_DIM; ++v)
                temp[v] = cell->vertex(v+1)->info().density() - denBase;
            matrixMultiplication(posMatInv, temp, denGrad);
            linearProfile = true;
        }

        // Constant scalar gradient across this cell.
#ifdef SCALAR
        Real sGrad[NO_DIM][noScalarComp];
        if (field.scalar || field.scalar_gradient)
        {
            Real temp[NO_DIM][noScalarComp];
            for (int v = 0; v < NO_DIM; ++v)
                for (size_t i = 0; i < noScalarComp; ++i)
                    temp[v][i] = cell->vertex(v+1)->info().myScalar()[i] - cell->vertex(0)->info().myScalar()[i];
            matrixMultiplication<noScalarComp>(posMatInv, temp, sGrad);
        }
#endif

        // ===================== Mass-conserving deposit of this tetrahedron =====================
        // PASS 1: gather every sub-sample point (nSub^NO_DIM per grid cell in the Eulerian bbox) that
        // lies inside the Eulerian simplex, recording its grid cell (flatIdx) and offset rel from
        // vertex 0. Sub-sampling probes the simplex volume uniformly.
        insideFlat.clear();
        insideRel.clear();
        insideW.clear();
#if NO_DIM==2
        for (int gi = iMin[0]; gi < iMax[0]; ++gi)
        for (int gj = iMin[1]; gj < iMax[1]; ++gj)
        {
            int wgi = userOptions.periodic ? ((gi % (int)nGrid[0] + (int)nGrid[0]) % (int)nGrid[0]) : gi;
            int wgj = userOptions.periodic ? ((gj % (int)nGrid[1] + (int)nGrid[1]) % (int)nGrid[1]) : gj;
            int gridIdx[2] = {wgi, wgj};
            int rawIdx[2]  = {gi, gj};
#elif NO_DIM==3
        for (int gi = iMin[0]; gi < iMax[0]; ++gi)
        for (int gj = iMin[1]; gj < iMax[1]; ++gj)
        for (int gk = iMin[2]; gk < iMax[2]; ++gk)
        {
            int wgi = userOptions.periodic ? ((gi % (int)nGrid[0] + (int)nGrid[0]) % (int)nGrid[0]) : gi;
            int wgj = userOptions.periodic ? ((gj % (int)nGrid[1] + (int)nGrid[1]) % (int)nGrid[1]) : gj;
            int wgk = userOptions.periodic ? ((gk % (int)nGrid[2] + (int)nGrid[2]) % (int)nGrid[2]) : gk;
            int gridIdx[3] = {wgi, wgj, wgk};
            int rawIdx[3]  = {gi, gj, gk};
#endif
            // flat index within this partition's sub-grid; inSub guard skips cells outside it.
            size_t flatIdx = 0;
            bool inSub = true;
            for (int d = 0; d < NO_DIM; ++d)
            {
                long loc = (long)gridIdx[d] - (long)subOrigin[d];
                if (loc < 0 || loc >= (long)subDims[d]) { inSub = false; break; }
                flatIdx = flatIdx * subDims[d] + (size_t)loc;
            }
            if (!inSub) continue;

            for (size_t sIdx = 0; sIdx < nSamplesPerCell; ++sIdx)
            {
                Real gridPos[NO_DIM];
                size_t rem = sIdx;
                for (int d = 0; d < NO_DIM; ++d)
                {
                    Real frac = (nSub == 1) ? Real(0.5)
                              : ( Real(int(rem % size_t(nSub))) + Real(0.5) ) / Real(nSub);
                    rem /= size_t(nSub);
                    gridPos[d] = boxCoordinates[2*d] + (rawIdx[d] + frac) * dx[d];
                }

                // Point-in-Eulerian-simplex test, relative to (wrapped) vertex 0. Ax stores edges as
                // rows, so bary = (Ax^-1)^T * rel: transpose posMatInv here (gradients use it directly).
                Real rel[NO_DIM];
                for (int d = 0; d < NO_DIM; ++d)
                    rel[d] = gridPos[d] - eulerPos[0][d];
                Real baryCoords[NO_DIM];
                for (int v = 0; v < NO_DIM; ++v)
                {
                    baryCoords[v] = Real(0.);
                    for (int i = 0; i < NO_DIM; ++i)
                        baryCoords[v] += posMatInv[i][v] * rel[i];
                }
                Real sum = 0.;
                bool inside = true;
                for (int i = 0; i < NO_DIM; ++i)
                {
                    if (baryCoords[i] < Real(-1.e-6)) { inside = false; break; }
                    sum += baryCoords[i];
                }
                if (!inside || sum > Real(1. + 1.e-6)) continue;

                insideFlat.push_back(flatIdx);
                for (int d = 0; d < NO_DIM; ++d) insideRel.push_back(rel[d]);
                if ( psLinear )
                {
                    // linear density at the sample (constant 1 for hull cells -> uniform);
                    // the +-1e-6 barycentric tolerance can graze negative: clamp
                    Real w = Real(1.);
                    if ( linearProfile )
                    {
                        w = denBase;
                        for (int d = 0; d < NO_DIM; ++d)
                            w += denGrad[d] * rel[d];
                        if ( w < Real(0.) ) w = Real(0.);
                    }
                    insideW.push_back(w);
                }
            }
        }

        // Fallback: a simplex smaller than the sub-sample spacing may catch no interior point. Deposit
        // its whole mass at the centroid's cell (the centroid is always inside) so no mass is ever lost
        // -- this is what makes even nSub==1 mass-conserving (no thin-stream undercount).
        bool useCentroid = false;
        size_t centroidFlat = 0;
        Real   centroidRel[NO_DIM];
        if (insideFlat.empty())
        {
            Real centroid[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d)
            {
                centroid[d] = Real(0.);
                for (int v = 0; v <= NO_DIM; ++v) centroid[d] += eulerPos[v][d];
                centroid[d] /= Real(NO_DIM + 1);
            }
            bool cValid = true;
            size_t f = 0;
            for (int d = 0; d < NO_DIM; ++d)
            {
                int raw = int(floor((centroid[d] - boxCoordinates[2*d]) / dx[d]));
                int w = userOptions.periodic ? ((raw % (int)nGrid[d] + (int)nGrid[d]) % (int)nGrid[d]) : raw;
                long loc = (long)w - (long)subOrigin[d];
                if (w < 0 || w >= (int)nGrid[d] || loc < 0 || loc >= (long)subDims[d]) { cValid = false; break; }
                f = f * subDims[d] + (size_t)loc;
            }
            if (!cValid) { continue; }   // simplex centroid outside this grid/partition region: drop it
            useCentroid = true;
            centroidFlat = f;
            for (int d = 0; d < NO_DIM; ++d) centroidRel[d] = centroid[d] - eulerPos[0][d];
        }

        // PASS 2: split tetMass among the interior samples and deposit. Equal shares (default)
        // reproduce the mass distribution across the cells the simplex covers; --ps-linear-deposit
        // weights the shares by the linear density profile instead, RENORMALIZED so the total is
        // exactly tetMass either way (at any nSub and for any V_eul) -- no cell can receive more
        // than the tetrahedron's mass and the caustic fix is untouched.
        size_t const nInside = useCentroid ? size_t(1) : insideFlat.size();
        Real   const uniformShare = tetMass / Real(nInside);
        double shareFactor = 0.;
        bool useLinearShares = false;
        if ( psLinear && !useCentroid )
        {
            double sumW = 0.;
            for (size_t s = 0; s < nInside; ++s) sumW += double(insideW[s]);
            if ( sumW > 0. )
            {
                shareFactor = double(tetMass) / sumW;
                useLinearShares = true;
            }
        }
        for (size_t s = 0; s < nInside; ++s)
        {
            size_t flatIdx;
            Real rel[NO_DIM];
            if (useCentroid)
            {
                flatIdx = centroidFlat;
                for (int d = 0; d < NO_DIM; ++d) rel[d] = centroidRel[d];
            }
            else
            {
                flatIdx = insideFlat[s];
                for (int d = 0; d < NO_DIM; ++d) rel[d] = insideRel[s*NO_DIM + d];
            }
            Real const massShare = useLinearShares ? Real(double(insideW[s]) * shareFactor)
                                                   : uniformShare;

            // density accumulates MASS here; it is divided by the cell volume once at the end.
            if (field.density) quantities->density[flatIdx] += massShare;
            if (needWeight && !weightIsDensity) massWeight[flatIdx] += massShare;

            // Mass-weighted velocity moment sum(m_s v_s); also needed for the dispersion (uses <v>).
            Pvector<Real,noVelComp> velVal;
            if (haveVel)
            {
                velVal = cell->vertex(0)->info().velocity();
                for (int i = 0; i < NO_DIM; ++i)
                    for (size_t j = 0; j < noVelComp; ++j)
                        velVal[j] += velGrad[i][j] * rel[i];
                quantities->velocity[flatIdx] += velVal * massShare;
            }

            if (field.velocity_gradient)
            {
                Pvector<Real,noGradComp> grad;
                for (size_t j = 0; j < noVelComp; ++j)
                    for (int i = 0; i < NO_DIM; ++i)
                        grad[j*NO_DIM+i] = velGrad[i][j];
                quantities->velocity_gradient[flatIdx] += grad * massShare;
            }

            // velocity dispersion: mass-weighted second moment sum(m_s v_i v_j) (upper triangle).
            // After normalization -> sigma_ij = <v_i v_j> - <v_i><v_j>.
            if (field.velocity_dispersion)
            {
                size_t c = 0;
                for (int i = 0; i < NO_DIM; ++i)
                    for (int j = i; j < NO_DIM; ++j)
                        quantities->velocity_dispersion[flatIdx][c++] += massShare * velVal[i] * velVal[j];
            }

#ifdef SCALAR
            if (field.scalar)
            {
                Pvector<Real,noScalarComp> scalarVal = cell->vertex(0)->info().myScalar();
                for (int i = 0; i < NO_DIM; ++i)
                    for (size_t j = 0; j < noScalarComp; ++j)
                        scalarVal[j] += sGrad[i][j] * rel[i];
                quantities->scalar[flatIdx] += scalarVal * massShare;
            }
            if (field.scalar_gradient)
            {
                Pvector<Real,noScalarGradComp> sgrad;
                for (size_t j = 0; j < noScalarComp; ++j)
                    for (int i = 0; i < NO_DIM; ++i)
                        sgrad[j*NO_DIM+i] = sGrad[i][j];
                quantities->scalar_gradient[flatIdx] += sgrad * massShare;
            }
#endif

            streamCount[flatIdx]++;
        }
    }   // end cell loop

    // Whichever deposit ran (GPU success above skipped the CPU loop; a GPU FAILURE fell back
    // to the CPU loop, which needed the triangulation), its last read of dt is behind us.
    // For the final call on an internally-owned dt, free the triangulation NOW -- during the
    // normalization below and the caller-side merge into the shared grids -- instead of at
    // partition scope end. ~650 B/vertex, ~5 GB per concurrent partition at production scale.
    if ( mayClearDT )
        dt.clear();

    if ( not userOptions.psSuppressGridStats )
    {
        message << MESSAGE::cGreen() << "Done.\n" << MESSAGE::cReset() << MESSAGE::Flush;
        if ( nDegenerateInverse > 0 )
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " dropped "
                    << MESSAGE::cMagenta() << nDegenerateInverse << MESSAGE::cReset()
                    << " degenerate (non-invertible) Eulerian cells.\n" << MESSAGE::Flush;
    }

    // turn density-weighted moments into mass-weighted means sum(rho_s f_s)/sum(rho_s). Serial: here,
    // BEFORE the density grid is converted to rho/rho_bar (it doubles as the weight when the density
    // field is selected -- see weightIsDensity above). deferNorm: hand un-normalized moments to the
    // caller, which sums across partitions and normalizes once (the only order correct for cells
    // spanning partitions); the weight travels as mass_weight, or is reconstructed from the summed
    // density by normalizePhaseSpace when it aliases the density grid.
    Real const invSamples = Real(1.) / Real(nSamplesPerCell);
    if ( needWeight )
    {
        if ( deferNorm )
        {
            if ( !weightIsDensity )
                quantities->mass_weight.swap( massWeight );   // sum(rho_s); normalized later by caller
        }
        else
            for (size_t i = 0; i < totalGrid; ++i)
            {
                Real const w = weightIsDensity ? quantities->density[i] : massWeight[i];
                if ( w > Real(0.) )
                {
                    Real const inv = Real(1.) / w;
                    if (haveVel)                 quantities->velocity[i]          *= inv;   // <v>
                    if (field.velocity_gradient) quantities->velocity_gradient[i] *= inv;
#ifdef SCALAR
                    if (field.scalar)            quantities->scalar[i]            *= inv;
                    if (field.scalar_gradient)   quantities->scalar_gradient[i]   *= inv;
#endif
                    if (field.velocity_dispersion)
                    {
                        // sigma_ij = <v_i v_j> - <v_i><v_j>; quantities->velocity is now <v>.
                        Pvector<Real,noVelComp> const &vbar = quantities->velocity[i];
                        size_t c = 0;
                        for (int a = 0; a < NO_DIM; ++a)
                            for (int b = a; b < NO_DIM; ++b)
                            {
                                Real s = quantities->velocity_dispersion[i][c] * inv - vbar[a]*vbar[b];
                                if (a == b and s < Real(0.)) s = Real(0.);   // variance: clamp FP-noise negatives
                                quantities->velocity_dispersion[i][c] = s;
                                ++c;
                            }
                    }
                }
            }
    }

    // density accumulated MASS (sum of per-tetrahedron mass shares); convert to a density by dividing
    // by the grid-cell volume, then normalize by averageDensity so the written field is rho/rho_bar
    // (mean-normalized "density contrast + 1", the SAME convention as standard DTFE -- since
    // 2026-07: older outputs stored the physical mass density; dtfelib.FieldSet detects both).
    // Mass conservation makes the box mean exactly 1 for any nSub. Stream count is averaged over
    // the nSub^NO_DIM sub-samples below.
    if ( field.density )
    {
        Real const invRhoBar = userOptions.averageDensity > Real(0.)
                             ? Real(1.) / userOptions.averageDensity : Real(1.);
        for (size_t i = 0; i < totalGrid; ++i)
            quantities->density[i] *= invRhoBar / cellVolume;
    }

    // multi-stream / coverage statistics. Suppressed in the partition path (each partition covers
    // only part of the grid); DTFE.cpp reports one aggregate from the summed grid.
    if ( not userOptions.psSuppressGridStats )
    {
        Real maxStreams = Real(0.);
        size_t multiStreamCells = 0;
        size_t coveredCells = 0;
        for (size_t i = 0; i < totalGrid; ++i)
        {
            Real const avg = Real(streamCount[i]) * invSamples;        // mean streams over the cell
            if (avg > maxStreams) maxStreams = avg;
            if (streamCount[i] > (int)nSamplesPerCell) multiStreamCells++;  // cell average > 1
            if (streamCount[i] > 0) coveredCells++;
        }
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " Max streams at a grid point: "
                << MESSAGE::cMagenta() << maxStreams << MESSAGE::cReset()
                << ", grid points with multi-stream: " << MESSAGE::cMagenta() << multiStreamCells
                << " (" << (100.*multiStreamCells/totalGrid) << "\%)" << MESSAGE::cReset() << "\n" << MESSAGE::Flush;
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " grid coverage: "
                << MESSAGE::cMagenta() << coveredCells << "/" << totalGrid
                << " (" << (100.*coveredCells/totalGrid) << "\%)" << MESSAGE::cReset()
                << ( coveredCells < totalGrid ?
                     "  -- uncovered cells suggest insufficient padding or a grid finer than the tessellation" : "" )
                << "\n" << MESSAGE::Flush;
    }

    // Export the per-grid-point stream count, averaged over the sub-samples.
    quantities->stream_count.resize(totalGrid);
    for (size_t i = 0; i < totalGrid; ++i)
        quantities->stream_count[i] = Real(streamCount[i]) * invSamples;

    // Record the sub-grid box so the caller (addFromSubgrid) can map results back into the full shared grid.
    for (int d = 0; d < NO_DIM; ++d)
    {
        quantities->ps_subOrigin[d] = subOrigin[d];
        quantities->ps_subDims[d]   = subDims[d];
    }
}

#endif // PHASE_SPACE
