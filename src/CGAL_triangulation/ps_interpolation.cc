/* Phase-space DTFE grid interpolation. Triangulation is built in Lagrangian space;
   Eulerian simplices can overlap (multi-stream), so we iterate over all cells and
   accumulate each simplex's contribution to every grid point it contains. */

#ifdef PHASE_SPACE

// PS-DTFE is validated only in 3D; the NO_DIM==2 path compiles but is untested.
// (#pragma message, not #warning: the build's global -Wno-cpp silences #warning directives)
#if NO_DIM==2
#pragma message "PS-DTFE (PHASE_SPACE) is not validated for NO_DIM==2; the 2D path is untested."
#endif

#include "triangulation_common.h"

#include <cstring>   // memcpy in the in-place cost-sort permutation

#if defined(PS_GPU) && NO_DIM==3
#include "gpu_host.h"   // GPU deposit (METAL=1 build); CPU deposit remains the fallback
#endif

#if NO_DIM==3
// --ps-exact-deposit: analytic tetrahedron-cell intersection moments (Powell & Abel 2015).
// Vendored C library; r3d.h carries its own extern "C" guard. 3D only.
#include "../../third_party/r3d/r3d.h"
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
    // so the box covers every written cell). off -> full grid.
    //
    // The box may WRAP a periodic axis: subOrigin+subDims can exceed nGrid, and the local index is
    // then (g - subOrigin + nGrid) % nGrid. Every consumer must apply that same wrap -- the CPU
    // scatter loops below, the GPU kernels (metal/ps_deposit.metal) and the
    // merge-back (Quantities::addFromSubgrid). The wrap is a no-op for an unwrapped box, so the
    // one expression serves both.
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
            PSCellGeometry geo;   // dummy/ownership/hull/degeneracy filters (no inverse check here)
            if ( !psFilterCell(cell, userOptions, boxCoordinates, false, NULL, geo) ) continue;
            Real (&ep)[NO_DIM+1][NO_DIM] = geo.eulerPos;
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
            else if (touched[d][0] && touched[d][nGrid[d]-1])
            {
                // Touched planes reach both ends. Under --periodic that is the NORMAL case for
                // any partition whose Eulerian footprint straddles the box seam, and keeping the
                // full axis here was what made every boundary-touching Lagrangian slab allocate
                // the whole grid (the documented "each partition writes the WHOLE Eulerian grid").
                // The footprint is really a WRAPPED arc, so crop to it: its complement is the
                // largest run of untouched planes, and because both ends are touched that run is
                // strictly interior and a plain linear scan finds it.
                int bestLen=0, bestStart=-1, curStart=-1;
                for (int g=0; g<(int)nGrid[d]; ++g)
                {
                    if (!touched[d][g]) { if (curStart<0) curStart=g; }
                    else if (curStart>=0)
                    {
                        int const len = g - curStart;
                        if (len > bestLen) { bestLen=len; bestStart=curStart; }
                        curStart = -1;
                    }
                }
                if (bestLen<=0) { subOrigin[d]=0; subDims[d]=nGrid[d]; }                       // every plane touched: genuinely full
                else { subOrigin[d]=(size_t)((bestStart+bestLen)%(int)nGrid[d]); subDims[d]=(size_t)((int)nGrid[d]-bestLen); }
            }
            else { subOrigin[d]=(size_t)lo; subDims[d]=(size_t)(hi-lo+1); }                    // crop to the touched [lo,hi] span
        }
    }
    size_t subTotal = 1; for (int d = 0; d < NO_DIM; ++d) subTotal *= subDims[d];
    totalGrid = subTotal;   // all allocations and loops below use the sub-grid size

    // --ps-vertex-mass: chart-independent per-tetrahedron masses. Count, per vertex, the
    // incident cells that will actually DEPOSIT; each tet's mass is then the sum of its
    // vertices' weight/degree shares (computed where tetMass is needed below). Mass follows
    // the particles instead of rho_bar * V_lag, which removes the 1 - D(z_ic)/D(z) contrast
    // suppression when the Lagrangian input is a PERTURBED IC snapshot (see --ps-vertex-mass
    // help). Counted on THIS triangulation with a reset first -- the unaveraged and averaged
    // passes share one dt, so the pass must be idempotent.
    //
    // The degree applies the SAME chart-independent drop filters as the deposit loop (dummy,
    // periodic hull artefact, singular Eulerian inverse) so that sum over deposited tets of
    // sum_v m_v/deg(v) equals sum_v m_v EXACTLY (up to vertices whose every incident tet is
    // dropped): counting a dropped tet in the degree used to send its vertices' shares into
    // the void with it -- a mass deficit tracking the dropped-tet fraction (0.74% at TNG z=0).
    // Ownership is deliberately SKIPPED: it tiles cells across partitions (each cell deposits
    // exactly once globally, and padded-region vertices see their full local neighborhood, so
    // per-partition degrees match the global tessellation).
    bool const psVertexMass = userOptions.psVertexMass;
    if ( psVertexMass )
    {
        for (DT::Finite_vertices_iterator vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit)
            vit->info().psDegree() = 0;
        size_t degDropped = 0;   // local counter: the deposit loop reports its own
        PS_FOREACH_FINITE_CELL
        {
            Cell_handle cell = itC;
            PSCellGeometry dgeo;
            if ( !psFilterCell(cell, userOptions, boxCoordinates, true, &degDropped, dgeo,
                               /*skipOwnership=*/true) )
                continue;
            for (int v = 0; v <= NO_DIM; ++v)
                cell->vertex(v)->info().psDegree() += 1;
        }
    }
    // per-tet vertex-share mass: sum over the cell's vertices of weight/degree
    auto psVertexShareMass = [](Cell_handle const &cell) -> double
    {
        double m = 0.;
        for (int v = 0; v <= NO_DIM; ++v)
        {
            int32_t const deg = cell->vertex(v)->info().psDegree();
            if (deg > 0)
                m += double(cell->vertex(v)->info().weight()) / double(deg);
        }
        return m;
    };

    // Allocate the requested output fields, zeroed since the cell loop accumulates into them.
    if ( field.density )
        quantities->density.assign(totalGrid, Real(0.));
    // Dispersion needs <v>, so allocate velocity storage even when only dispersion was
    // requested -- EXCEPT under --ps-volume-weighted, where the dispersion carries its OWN
    // mass-weighted mean (disp_velocity/disp_weight below): a dispersion-only volume-weighted
    // run would otherwise allocate, deposit and GPU-round-trip a 12 B/cell grid nothing reads.
    bool const haveVel = field.velocity
                         || (field.velocity_dispersion && !userOptions.psVolumeWeighted);
    // the dispersion always needs the per-sample velocity VALUES (for sum m v_i v_j and its
    // own mean), even when the velocity GRID above is gated away -- keep the two distinct
    bool const needVelValues = haveVel || field.velocity_dispersion;
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
    // Sampled deposit: incremented once per in-simplex sample, divided by nSub^NO_DIM at the end
    // -> the cell-mean stream multiplicity. Exact deposit: this counts TETRAHEDRA TOUCHING the
    // cell (every nonzero clip, including corner grazes) -- a geometric multiplicity that is NOT
    // a stream count (it runs to hundreds), so it is exported separately as '.tetTouch' and the
    // physical multiplicity is accumulated in exactMult below.
    std::vector<int> streamCount(totalGrid, 0);
    // --ps-exact-deposit: exact cell-mean stream multiplicity = (1/V_cell) * sum_tets V_int,
    // the analytic nSub->infinity limit of the sampled definition (which is the sample-mean
    // multiplicity). On the same scale as the sampled '.streams', so single-stream masking
    // works under either deposit -- but ONLY to a tolerance (|s-1| < 1e-3, see
    // dtfelib.STREAM_TOL): this is a float sum, so single-stream cells land on 1.0 +/- eps,
    // never exactly 1. Grazing slivers contribute ~0 here instead of a full +1, which also
    // makes it far more robust than the raw touch count (float GPU included).
    std::vector<Real> exactMult;
    if ( userOptions.psExactDeposit ) exactMult.assign(totalGrid, Real(0.));

    // --ps-caustics: per-cell orientation mask (bit0 = a det(Ax)>0 stream overlaps the cell,
    // bit1 = det<0). det(Ax)'s sign is the parity of the Lagrangian->Eulerian map (CGAL cells
    // are positively oriented in Lagrangian space), which flips at every fold -- both bits set
    // means a fold caustic surface crosses the cell. Exported as small exact ints in Real.
    bool const psCaustics = userOptions.psCaustics;
    std::vector<unsigned char> orientBits;
    if ( psCaustics )
        orientBits.assign(totalGrid, 0);

    // --ps-halo-release: threshold D on the geometric stream density rho_geo/rho_bar =
    // V_lag/V_eul = |det Lag| / |det Ax| (the 1/NO_DIM! factors cancel). Tets beyond D sit
    // deep inside virialized, mixed regions where fold counts explode and the sheet estimate
    // is unconverged anyway (Stuecker et al. 2021): they take the mass-conserving centroid
    // fallback instead of the bbox rasterization. Classified in double in BOTH deposits
    // (partition protocol: CPU and GPU must keep/release identical tets). 0 = off.
    double const psHaloRelease = double(userOptions.psHaloRelease);
    size_t nReleased = 0;

    // --ps-exact-deposit: analytic per-cell intersection moments instead of nSub^3 sampling.
    // The exact deposit is the nSub->infinity limit, so nSub is ignored. '.streams' carries
    // the volume-weighted multiplicity (see exactMult above): a FLOAT, so the partition merge
    // is NOT bit-exact -- the per-tet volume shares sum in thread order. The raw integer
    // tet-touch count, which does merge bit-exactly, is exported separately as '.tetTouch'.
    bool const psExact = userOptions.psExactDeposit;

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
    // --ps-volume-weighted: the velocity/scalar moments (and their normalizer) carry equal
    // EULERIAN-VOLUME shares instead of mass shares -> the normalized cell values are volume
    // averages (the standard-DTFE '_a' convention) instead of mass-weighted means. Density,
    // stream count and caustics always keep the mass deposit.
    bool const psVolWeighted = userOptions.psVolumeWeighted;
    // when the density field is selected, its accumulator receives exactly the same per-cell
    // mass sums as massWeight would (both add massShare at the same sites), so the weight
    // ALIASES the density grid instead of allocating a second full grid (4 B/cell). The
    // deferNorm caller reconstructs the weight from the summed density (normalizePhaseSpace
    // scale argument), since density is only converted to rho/rho_bar after weighting here.
    // Volume weighting breaks the aliasing (the moment normalizer is no longer the mass sum),
    // so it always allocates the explicit weight grid.
    bool const weightIsDensity = needWeight && field.density && !psVolWeighted;
    std::vector<Real> massWeight;
    if ( needWeight && !weightIsDensity ) massWeight.assign(totalGrid, Real(0.));
    // --ps-volume-weighted keeps the DISPERSION mass-weighted (sigma_ij is a moment of the
    // phase-space distribution f -- f-weighted by definition) while velocity/gradient go
    // volume-weighted. sigma_ij = <v_i v_j>_m - <v_i>_m<v_j>_m then needs its own mass-weighted
    // mean and normalizer, because 'velocity' now carries the volume-weighted one.
    bool const dispNeedsOwnWeight = psVolWeighted && field.velocity_dispersion;
    std::vector<Real> dispWeight;                      // sum(m_s)
    std::vector< Pvector<Real,noVelComp> > dispVel;    // sum(m_s v_s)
    if ( dispNeedsOwnWeight )
    {
        dispWeight.assign(totalGrid, Real(0.));
        dispVel.assign(totalGrid, Pvector<Real,noVelComp>::zero());
    }

    // Reusable per-tetrahedron scratch for the mass-conserving deposit (see the cell loop): the flat
    // grid index and Eulerian offset (rel = samplePos - vertex0) of every in-simplex sample point.
    std::vector<size_t> insideFlat;
    std::vector<Real>   insideRel;   // insideFlat.size() * NO_DIM, row-major per sample
    // --ps-linear-deposit: per-sample weight = the DTFE-interpolated linear density at the sample
    // (clamped >= 0); PASS 2 renormalizes the weights to the tet mass, so the deposited total
    // still equals tetMass exactly and mass conservation is untouched.
    bool const psLinear = userOptions.psLinearDeposit;
    std::vector<Real>   insideW;
    // --ps-exact-deposit per-tet scratch: 10 order-2 moments and the mass weight per hit cell
    std::vector<double> exMom, exW;

    if ( not userOptions.psSuppressGridStats )
        message << "\n" << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset()
                << " Interpolating fields to grid by iterating over all Delaunay cells ...\n" << MESSAGE::Flush;

    // GPU deposit (--ps-gpu, Metal build): extract flat per-tet arrays with EXACTLY the CPU
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
        warning << "--ps-gpu does not support the scalar fields; using the CPU deposit for this pass.\n" << MESSAGE::EndWarning;
        tryMetal = false;
    }
#endif
    // The kernels index cells with a 32-bit 'uint flat' (metal/ps_deposit.metal).
    // Above 2^32 cells it wraps -- and because the buffers are LARGER than 2^32, the wrapped index
    // is still in bounds: no crash, no OOB, just atomics landing in the wrong cells. Fall back to
    // the CPU instead, mirroring the standard-DTFE guard in averaged_interpolation_1.cc. This is
    // checked on subTotal, the only quantity that knows about the wrap-to-full-axis case above:
    // the cap is on the PARTITION's sub-box, so a finer --partition can bring a big grid back
    // under it. Uniform across field selections -- the moment offsets are already 64-bit.
    if ( tryMetal and subTotal > size_t(UINT32_MAX) )
    {
        MESSAGE::Warning warning( userOptions.verboseLevel );
        warning << "--ps-gpu does not support more than 2^32 cells per partition sub-grid (this one has "
                << subTotal << ", i.e. > " << size_t(UINT32_MAX) << "); using the CPU deposit for this pass. "
                << "A finer --partition would bring the sub-grid back under the cap.\n" << MESSAGE::EndWarning;
        tryMetal = false;
    }
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
        // --ps-halo-release: released tets never reach the GPU arrays; their single-cell
        // centroid deposits (same arithmetic as the CPU fallback) are collected here and
        // applied on the host AFTER the GPU grids are copied back (the copy ASSIGNS, so
        // these are the only further accumulations). Discarded if the GPU dispatch fails --
        // the CPU loop then re-handles every cell. momw = the moment weight (V_eul under
        // --ps-volume-weighted, else the mass); orient = the caustic orientation bits.
        struct ReleasedTet { size_t flat; Real mass; Real momw; Real volFrac; unsigned char orient; Real vel[noVelComp]; Real grad[noGradComp]; };
        std::vector<ReleasedTet> releasedTets;
        PS_FOREACH_FINITE_CELL
        {
            Cell_handle cell = itC;
            PSCellGeometry geo;   // the CPU loop's filters, including the zero-inverse check
            if ( !psFilterCell(cell, userOptions, boxCoordinates, true, &nDegenerateInverse, geo) ) continue;
            bool const hadBadVertex = geo.useVolumeRatioDensity;
            Real (&ep)[NO_DIM+1][NO_DIM] = geo.eulerPos;
            double Lag[NO_DIM][NO_DIM];
            for (int v=0;v<NO_DIM;++v) for (int i=0;i<NO_DIM;++i) Lag[v][i]=double(cell->vertex(v+1)->point()[i])-double(cell->vertex(0)->point()[i]);
            // absDetLag always computed: the --ps-halo-release classification below stays
            // geometric (V_lag/V_eul) so kept/released sets are identical under either mass
            double const absDetLag = std::fabs(determinant(Lag));
            float tm = psVertexMass ? float( psVertexShareMass(cell) )
                                    : float( double(userOptions.averageDensity)*absDetLag/factorial(NO_DIM) );
            if (tm<=0.f) continue;
            if ( psHaloRelease > 0. && absDetLag > psHaloRelease * geo.cellAbsDet )
            {
                // identical classification to the CPU loop (double |det Lag| > D*|det Ax|);
                // centroid cell + centroid-evaluated velocity mirror the CPU fallback exactly
                Real centroid[NO_DIM];
                for (int d = 0; d < NO_DIM; ++d)
                {
                    centroid[d] = Real(0.);
                    for (int v = 0; v <= NO_DIM; ++v) centroid[d] += ep[v][d];
                    centroid[d] /= Real(NO_DIM + 1);
                }
                bool cValid = true;
                size_t f = 0;
                for (int d = 0; d < NO_DIM; ++d)
                {
                    int raw = int(floor((centroid[d] - boxCoordinates[2*d]) / dx[d]));
                    int w = userOptions.periodic ? ((raw % (int)nGrid[d] + (int)nGrid[d]) % (int)nGrid[d]) : raw;
                    long loc = (long)w - (long)subOrigin[d];
                    if (loc < 0) loc += (long)nGrid[d];        // sub-box may wrap a periodic axis
                    if (w < 0 || w >= (int)nGrid[d] || loc < 0 || loc >= (long)subDims[d]) { cValid = false; break; }
                    f = f * subDims[d] + (size_t)loc;
                }
                if (!cValid) { continue; }   // centroid outside this grid/partition region: drop (as the CPU fallback does)
                ReleasedTet rt;
                rt.flat = f;
                rt.mass = Real(tm);
                rt.momw = psVolWeighted ? Real( geo.cellAbsDet / factorial(NO_DIM) ) : rt.mass;
                rt.volFrac = Real( geo.cellAbsDet / factorial(NO_DIM) / double(cellVolume) );
                rt.orient = geo.cellDet > 0. ? 1 : 2;
                if ( needsVelArrays )
                {
                    Real velGrad[NO_DIM][noVelComp], temp[NO_DIM][noVelComp];
                    for (int v = 0; v < NO_DIM; ++v)
                        for (size_t j = 0; j < noVelComp; ++j)
                            temp[v][j] = cell->vertex(v+1)->info().velocity(j) - cell->vertex(0)->info().velocity(j);
                    matrixMultiplication<noVelComp>(geo.posMatInv, temp, velGrad);
                    for (size_t j = 0; j < noVelComp; ++j)
                    {
                        rt.vel[j] = cell->vertex(0)->info().velocity(j);
                        for (int i = 0; i < NO_DIM; ++i)
                            rt.vel[j] += velGrad[i][j] * (centroid[i] - ep[0][i]);
                    }
                    if ( fGrad )
                        for (size_t j = 0; j < noVelComp; ++j)
                            for (int i = 0; i < NO_DIM; ++i)
                                rt.grad[j*NO_DIM+i] = velGrad[i][j];
                }
                releasedTets.push_back(rt);
                ++nReleased;
                continue;
            }
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
                                  userOptions.periodic, fVel, fDisp, fGrad, psLinear,
                                  psVolWeighted, psCaustics, psExact, gpuOut, metalErr) )
        {
            if ( not userOptions.psSuppressGridStats )
                message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " " << gpuBackendName() << " GPU deposit ("
                        << MESSAGE::cMagenta() << gpuDeviceName() << MESSAGE::cReset() << "), "
                        << MESSAGE::cMagenta() << nTetsDeposited << MESSAGE::cReset() << " tetrahedra.\n" << MESSAGE::Flush;
            for (size_t i = 0; i < totalGrid; ++i)
            {
                if (field.density) quantities->density[i] = Real(gpuOut.mass[i]);
                // moment normalizer: the volume-share sum under --ps-volume-weighted, else the mass
                if (needWeight && !weightIsDensity)
                    massWeight[i] = Real( psVolWeighted ? gpuOut.momw[i] : gpuOut.mass[i] );
                if (haveVel)
                    for (size_t j = 0; j < noVelComp; ++j)
                        quantities->velocity[i][j] = Real(gpuOut.mom[i*3+j]);
                if (field.velocity_gradient)
                    for (size_t q = 0; q < noGradComp; ++q)
                        quantities->velocity_gradient[i][q] = Real(gpuOut.grad[i*9+q]);
                if (field.velocity_dispersion)
                    for (size_t cc = 0; cc < noDispComp; ++cc)
                        quantities->velocity_dispersion[i][cc] = Real(gpuOut.m2[i*6+cc]);
                if (dispNeedsOwnWeight)
                {   // the dispersion's own mass-weighted mean/normalizer (kernel buffers 13/14)
                    dispWeight[i] = Real(gpuOut.dispw[i]);
                    for (size_t j = 0; j < noVelComp; ++j)
                        dispVel[i][j] = Real(gpuOut.dispvel[i*3+j]);
                }
                streamCount[i] = int(gpuOut.streams[i]);
                if (psExact)
                    exactMult[i] = Real(gpuOut.streamvol[i]);   // exact cell-mean multiplicity
                if (psCaustics)
                    orientBits[i] = (unsigned char)(gpuOut.caustic[i] & 3u);
            }
            // --ps-halo-release: host-side monolithic centroid deposits of the released tets
            // (same accumulation sites and moments as the CPU fallback's single sample)
            for (ReleasedTet const &rt : releasedTets)
            {
                if (field.density) quantities->density[rt.flat] += rt.mass;
                if (needWeight && !weightIsDensity) massWeight[rt.flat] += rt.momw;
                if (haveVel)
                    for (size_t j = 0; j < noVelComp; ++j)
                        quantities->velocity[rt.flat][j] += rt.vel[j] * rt.momw;
                if (field.velocity_gradient)
                    for (size_t q = 0; q < noGradComp; ++q)
                        quantities->velocity_gradient[rt.flat][q] += rt.grad[q] * rt.momw;
                if (field.velocity_dispersion)
                {
                    size_t c = 0;
                    for (int a = 0; a < NO_DIM; ++a)
                        for (int b = a; b < NO_DIM; ++b)
                            quantities->velocity_dispersion[rt.flat][c++] += rt.mass * rt.vel[a] * rt.vel[b];
                    if ( dispNeedsOwnWeight )
                    {
                        dispWeight[rt.flat] += rt.mass;
                        for (size_t j = 0; j < noVelComp; ++j)
                            dispVel[rt.flat][j] += rt.vel[j] * rt.mass;
                    }
                }
                streamCount[rt.flat]++;
                if (psExact)
                    exactMult[rt.flat] += rt.volFrac;   // V_tet / V_cell, as the CPU fallback deposits
                if (psCaustics)
                    orientBits[rt.flat] |= rt.orient;
            }
            metalDeposited = true;
        }
        else
        {
            MESSAGE::Warning warning( userOptions.verboseLevel );
            warning << "PS-DTFE GPU deposit unavailable (" << metalErr << "); using the CPU deposit.\n" << MESSAGE::EndWarning;
            nReleased = 0;   // releasedTets are discarded; the CPU loop below recounts them
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
            warning << "--ps-gpu requested but this binary was built without GPU support (rebuild with METAL=1; or NO_DIM!=3); using the CPU deposit.\n" << MESSAGE::EndWarning;
        }
    }
#endif // PS_GPU

    // per-cell scatter is serial; parallelism is one level up over Lagrangian partitions (DTFE.cpp)
    if ( !metalDeposited )
    PS_FOREACH_FINITE_CELL
    {
        Cell_handle cell = itC;

        // Shared filter chain (ps_cell_filter.h): dummy-vertex skip, Lagrangian-centroid
        // ownership, hull handling, minimum-image wrap, degeneracy + zero-inverse checks.
        PSCellGeometry geo;
        if ( !psFilterCell(cell, userOptions, boxCoordinates, true, &nDegenerateInverse, geo) )
            continue;
        bool const useVolumeRatioDensity = geo.useVolumeRatioDensity;
        Real (&eulerPos)[NO_DIM+1][NO_DIM] = geo.eulerPos;
        Real (&posMatInv)[NO_DIM][NO_DIM] = geo.posMatInv;

        // PS-DTFE per-tetrahedron MASS (Abel/Hahn/Shandarin tetrahedra method): each Lagrangian flow
        // element carries a constant mass m = averageDensity * V_lag, conserved as it maps to its
        // present-day Eulerian simplex (Eulerian volume V_eul). We deposit this finite MASS onto the
        // grid (mass-conserving; see the scatter below), instead of painting the per-tetrahedron
        // density rho = m / V_eul at grid points: at a caustic V_eul -> 0 so rho -> ~1e7, and
        // point-sampling that density either misses the thin simplex (mass undercounted) or splats
        // ~1e7 over a whole grid cell (the grid-aligned "square" over-densities, worse under
        // sub-sampling because more sample points fall in the cell). Depositing m is exact at any nSub.
        Real tetMass;
        double absDetLag;
        {
            double Lag[NO_DIM][NO_DIM];
            for (int v = 0; v < NO_DIM; ++v)
                for (int i = 0; i < NO_DIM; ++i)
                    Lag[v][i] = double(cell->vertex(v+1)->point()[i]) - double(cell->vertex(0)->point()[i]);
            // absDetLag always computed: the --ps-halo-release classification stays geometric
            // (V_lag/V_eul) so kept/released sets are identical under either mass convention
            absDetLag = std::fabs(determinant(Lag));
            tetMass = psVertexMass ? Real( psVertexShareMass(cell) )
                                   : Real( double(userOptions.averageDensity) * absDetLag / factorial(NO_DIM) );  // = rho_bar * V_lag
            if (tetMass < Real(0.)) tetMass = Real(0.);
        }

        // --ps-halo-release: released tets skip PASS 1 below (zero-trip window), so the
        // empty sample set drives the mass-conserving centroid fallback -- a monolithic
        // single-cell deposit with the centroid-evaluated velocity.
        bool const released = psHaloRelease > 0. && absDetLag > psHaloRelease * geo.cellAbsDet;
        if (released) ++nReleased;

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
            // sub-sample points can lie in edge cells; the exact deposit clips the bbox
            // window directly (no sub-samples), so the expansion would only add empty clips
            if (nSub > 1 && !psExact) { iMin[d] -= 1; iMax[d] += 1; }
            if (!userOptions.periodic)
            {
                if (iMin[d] < 0) iMin[d] = 0;
                if (iMax[d] > (int)nGrid[d]) iMax[d] = (int)nGrid[d];
                if (iMin[d] >= (int)nGrid[d] || iMax[d] <= 0) { outsideGrid = true; break; }
            }
        }
        if (outsideGrid) { continue; }
        if (released)
            for (int d = 0; d < NO_DIM; ++d) iMax[d] = iMin[d];   // zero-trip PASS 1 -> centroid fallback

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

#if NO_DIM==3
        // ===================== exact conservative deposit (--ps-exact-deposit) =====================
        // r3d (Powell & Abel 2015): clip the Eulerian tetrahedron against every grid cell in
        // its bbox window and integrate the moments analytically. All geometry lives in
        // coordinates RELATIVE to (wrapped) vertex 0 -- the same affine frame as velGrad and
        // denGrad, so the moments feed the linear profiles directly. Released tets (--ps-halo-
        // release) fall through to the sampled path below, whose zero-tripped window drives
        // the same monolithic centroid fallback.
        if ( psExact && !released )
        {
            r3d_rvec3 tv[NO_DIM+1];
            for (int v = 0; v <= NO_DIM; ++v)
                for (int d = 0; d < NO_DIM; ++d)
                    tv[v].xyz[d] = double(eulerPos[v][d]) - double(eulerPos[0][d]);
            // r3d_init_tet wants positive orientation; a folded (negative-parity) tet is
            // handed over with vertices 1,2 swapped -- vertex 0 stays the frame origin
            if ( geo.cellDet < 0. )
            {
                r3d_rvec3 const tmp = tv[1]; tv[1] = tv[2]; tv[2] = tmp;
            }
            r3d_poly tetPoly;
            r3d_init_tet(&tetPoly, tv);

            // PASS 1 (exact): moments of tet ∩ cell for every window cell with nonzero volume.
            // Order 2 = 10 moments [1, x, y, z, x2, xy, xz, y2, yz, z2] in the vertex-0 frame.
            insideFlat.clear();          // reused scratch: flat sub-grid index per hit cell
            exMom.clear();
            exW.clear();
            double sumW = 0.;            // per-tet weight normalizer (uniform: volume)
            for (int gi = iMin[0]; gi < iMax[0]; ++gi)
            for (int gj = iMin[1]; gj < iMax[1]; ++gj)
            for (int gk = iMin[2]; gk < iMax[2]; ++gk)
            {
                int const wgi = userOptions.periodic ? ((gi % (int)nGrid[0] + (int)nGrid[0]) % (int)nGrid[0]) : gi;
                int const wgj = userOptions.periodic ? ((gj % (int)nGrid[1] + (int)nGrid[1]) % (int)nGrid[1]) : gj;
                int const wgk = userOptions.periodic ? ((gk % (int)nGrid[2] + (int)nGrid[2]) % (int)nGrid[2]) : gk;
                int const gridIdx[3] = {wgi, wgj, wgk};
                int const rawIdx[3]  = {gi, gj, gk};
                size_t flatIdx = 0;
                bool inSub = true;
                for (int d = 0; d < NO_DIM; ++d)
                {
                    long loc = (long)gridIdx[d] - (long)subOrigin[d];
                    if (loc < 0) loc += (long)nGrid[d];        // sub-box may wrap a periodic axis
                    if (loc < 0 || loc >= (long)subDims[d]) { inSub = false; break; }
                    flatIdx = flatIdx * subDims[d] + (size_t)loc;
                }
                if (!inSub) continue;

                // cell bounds in the vertex-0 frame (RAW indices: the periodic image the
                // wrapped tet actually intersects), then 6 clip planes n.x + d >= 0
                r3d_plane planes[6];
                for (int d = 0; d < NO_DIM; ++d)
                {
                    double const lo = double(boxCoordinates[2*d]) + rawIdx[d] * double(dx[d]) - double(eulerPos[0][d]);
                    double const hi = lo + double(dx[d]);
                    for (int q = 0; q < 2; ++q)
                    {
                        planes[2*d+q].n.xyz[0] = 0.; planes[2*d+q].n.xyz[1] = 0.; planes[2*d+q].n.xyz[2] = 0.;
                    }
                    planes[2*d].n.xyz[d]   =  1.; planes[2*d].d   = -lo;
                    planes[2*d+1].n.xyz[d] = -1.; planes[2*d+1].d =  hi;
                }
                r3d_poly piece = tetPoly;
                r3d_clip(&piece, planes, 6);
                if ( piece.nverts == 0 ) continue;
                r3d_real mom[10];
                r3d_reduce(&piece, mom, 2);
                if ( !(mom[0] > 0.) ) continue;

                insideFlat.push_back(flatIdx);
                for (int m = 0; m < 10; ++m) exMom.push_back(double(mom[m]));
                // per-cell mass weight: exact volume, or the exact integral of the linear
                // density profile under --ps-linear-deposit (clamped like the sampled path)
                double w = mom[0];
                if ( psLinear && linearProfile )
                {
                    w = double(denBase) * mom[0];
                    for (int d = 0; d < NO_DIM; ++d)
                        w += double(denGrad[d]) * mom[1+d];
                    if ( w < 0. ) w = 0.;
                }
                exW.push_back(w);
                sumW += w;
            }

            // PASS 2 (exact): renormalize the shares to tetMass -- conservation is exact per
            // tet by construction (and matches the sampled path's semantics when part of a
            // non-periodic tet leaves the grid). An all-empty window (degenerate sliver below
            // r3d's resolution) falls through to the sampled path's centroid fallback.
            if ( sumW > 0. )
            {
                double const shareFac = double(tetMass) / sumW;
                for (size_t c = 0; c < insideFlat.size(); ++c)
                {
                    size_t const flatIdx = insideFlat[c];
                    double const *m = &exMom[c*10];
                    Real const massShare = Real( exW[c] * shareFac );
                    // moment weight: mass share, or the analytic intersection volume V_int
                    // (--ps-volume-weighted, the exact continuum volume share)
                    Real const momShare = psVolWeighted ? Real( m[0] ) : massShare;
                    if (field.density) quantities->density[flatIdx] += massShare;
                    if (needWeight && !weightIsDensity) massWeight[flatIdx] += momShare;

                    // exact mean of the linear velocity profile over this intersection
                    Pvector<Real,noVelComp> velBar;
                    if (needVelValues)
                    {
                        double const invV = 1. / m[0];
                        double cen[NO_DIM] = { m[1]*invV, m[2]*invV, m[3]*invV };
                        for (size_t j = 0; j < noVelComp; ++j)
                        {
                            double vb = double( cell->vertex(0)->info().velocity(j) );
                            for (int i = 0; i < NO_DIM; ++i)
                                vb += double(velGrad[i][j]) * cen[i];
                            velBar[j] = Real(vb);
                        }
                        if (haveVel)
                            quantities->velocity[flatIdx] += velBar * momShare;
                    }
                    if (field.velocity_gradient)
                    {
                        Pvector<Real,noGradComp> grad;
                        for (size_t j = 0; j < noVelComp; ++j)
                            for (int i = 0; i < NO_DIM; ++i)
                                grad[j*NO_DIM+i] = velGrad[i][j];
                        quantities->velocity_gradient[flatIdx] += grad * momShare;
                    }
                    if (field.velocity_dispersion)
                    {
                        // exact second moment of the linear profile: <v_i v_j> over the piece
                        // = vbar_i vbar_j + (G^T Cov G)_ij with Cov the piece's position
                        // covariance from the order-2 moments (M2 index map: xx xy xz yy yz zz)
                        double const invV = 1. / m[0];
                        double const cen[NO_DIM] = { m[1]*invV, m[2]*invV, m[3]*invV };
                        double cov[NO_DIM][NO_DIM];
                        cov[0][0] = m[4]*invV - cen[0]*cen[0];
                        cov[0][1] = cov[1][0] = m[5]*invV - cen[0]*cen[1];
                        cov[0][2] = cov[2][0] = m[6]*invV - cen[0]*cen[2];
                        cov[1][1] = m[7]*invV - cen[1]*cen[1];
                        cov[1][2] = cov[2][1] = m[8]*invV - cen[1]*cen[2];
                        cov[2][2] = m[9]*invV - cen[2]*cen[2];
                        size_t c2 = 0;
                        for (int a = 0; a < NO_DIM; ++a)
                            for (int b = a; b < NO_DIM; ++b)
                            {
                                double gcg = 0.;
                                for (int i = 0; i < NO_DIM; ++i)
                                    for (int k = 0; k < NO_DIM; ++k)
                                        gcg += double(velGrad[i][a]) * cov[i][k] * double(velGrad[k][b]);
                                quantities->velocity_dispersion[flatIdx][c2++] +=
                                    massShare * ( velBar[a] * velBar[b] + Real(gcg) );
                            }
                        if ( dispNeedsOwnWeight )
                        {
                            dispWeight[flatIdx] += massShare;
                            dispVel[flatIdx]    += velBar * massShare;
                        }
                    }
#ifdef SCALAR
                    if (field.scalar)
                    {
                        // exact mean of the linear scalar profile (same construction as velBar)
                        double const invV = 1. / m[0];
                        double const cen[NO_DIM] = { m[1]*invV, m[2]*invV, m[3]*invV };
                        Pvector<Real,noScalarComp> scalarVal;
                        for (size_t j = 0; j < noScalarComp; ++j)
                        {
                            double sb = double( cell->vertex(0)->info().myScalar()[j] );
                            for (int i = 0; i < NO_DIM; ++i)
                                sb += double(sGrad[i][j]) * cen[i];
                            scalarVal[j] = Real(sb);
                        }
                        quantities->scalar[flatIdx] += scalarVal * momShare;
                    }
                    if (field.scalar_gradient)
                    {
                        Pvector<Real,noScalarGradComp> sgrad;
                        for (size_t j = 0; j < noScalarComp; ++j)
                            for (int i = 0; i < NO_DIM; ++i)
                                sgrad[j*NO_DIM+i] = sGrad[i][j];
                        quantities->scalar_gradient[flatIdx] += sgrad * momShare;
                    }
#endif
                    streamCount[flatIdx]++;   // raw tet-touch count -> '.tetTouch'
                    exactMult[flatIdx] += Real( m[0] / double(cellVolume) );   // exact multiplicity share
                    if ( psCaustics )
                        orientBits[flatIdx] |= (geo.cellDet > 0. ? 1 : 2);
                }
                continue;   // next tetrahedron (the sampled PASS 1/2 below is skipped)
            }
        }
#endif  // NO_DIM==3 (exact deposit)

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
                if (loc < 0) loc += (long)nGrid[d];            // sub-box may wrap a periodic axis
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
                if (loc < 0) loc += (long)nGrid[d];        // sub-box may wrap a periodic axis
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
        // --ps-volume-weighted: equal Eulerian-volume share per sample for the velocity
        // moments (the centroid fallback carries the whole tet volume); density keeps mass
        Real   const volumeShare  = Real( geo.cellAbsDet / factorial(NO_DIM) ) / Real(nInside);
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
            // moment weight: mass share (default, momentum-like means) or equal volume share
            // (--ps-volume-weighted, volume-average means); density always deposits MASS
            Real const momShare = psVolWeighted ? volumeShare : massShare;

            // density accumulates MASS here; it is divided by the cell volume once at the end.
            if (field.density) quantities->density[flatIdx] += massShare;
            if (needWeight && !weightIsDensity) massWeight[flatIdx] += momShare;

            // Weighted velocity moment sum(w_s v_s); also needed for the dispersion (uses <v>).
            Pvector<Real,noVelComp> velVal;
            if (needVelValues)
            {
                velVal = cell->vertex(0)->info().velocity();
                for (int i = 0; i < NO_DIM; ++i)
                    for (size_t j = 0; j < noVelComp; ++j)
                        velVal[j] += velGrad[i][j] * rel[i];
                if (haveVel)
                    quantities->velocity[flatIdx] += velVal * momShare;
            }

            if (field.velocity_gradient)
            {
                Pvector<Real,noGradComp> grad;
                for (size_t j = 0; j < noVelComp; ++j)
                    for (int i = 0; i < NO_DIM; ++i)
                        grad[j*NO_DIM+i] = velGrad[i][j];
                quantities->velocity_gradient[flatIdx] += grad * momShare;
            }

            // velocity dispersion: weighted second moment sum(w_s v_i v_j) (upper triangle).
            // After normalization -> sigma_ij = <v_i v_j> - <v_i><v_j>.
            if (field.velocity_dispersion)
            {
                size_t c = 0;
                for (int i = 0; i < NO_DIM; ++i)
                    for (int j = i; j < NO_DIM; ++j)
                        quantities->velocity_dispersion[flatIdx][c++] += massShare * velVal[i] * velVal[j];
                if ( dispNeedsOwnWeight )
                {
                    dispWeight[flatIdx] += massShare;
                    dispVel[flatIdx]    += velVal * massShare;
                }
            }

#ifdef SCALAR
            if (field.scalar)
            {
                Pvector<Real,noScalarComp> scalarVal = cell->vertex(0)->info().myScalar();
                for (int i = 0; i < NO_DIM; ++i)
                    for (size_t j = 0; j < noScalarComp; ++j)
                        scalarVal[j] += sGrad[i][j] * rel[i];
                quantities->scalar[flatIdx] += scalarVal * momShare;
            }
            if (field.scalar_gradient)
            {
                Pvector<Real,noScalarGradComp> sgrad;
                for (size_t j = 0; j < noScalarComp; ++j)
                    for (int i = 0; i < NO_DIM; ++i)
                        sgrad[j*NO_DIM+i] = sGrad[i][j];
                quantities->scalar_gradient[flatIdx] += sgrad * momShare;
            }
#endif

            streamCount[flatIdx]++;
            // --ps-exact-deposit fall-through (empty clip window, released or sub-resolution
            // tets): the tet's volume is shared equally among its nInside samples, so each
            // carries V_tet/nInside -- for the monolithic centroid fallback (nInside==1) that
            // is exactly V_tet/V_cell, the same volume fraction the exact path deposits.
            if ( psExact )
                exactMult[flatIdx] += Real( geo.cellAbsDet / factorial(NO_DIM) / double(cellVolume) )
                                      / Real(nInside);
            if ( psCaustics )
                orientBits[flatIdx] |= (geo.cellDet > 0. ? 1 : 2);   // fold parity: sign of det(Ax)
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
        if ( nReleased > 0 )
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " released "
                    << MESSAGE::cMagenta() << nReleased << MESSAGE::cReset()
                    << " halo-interior tetrahedra (rho_geo/rho_bar > " << userOptions.psHaloRelease
                    << ") to monolithic centroid deposits.\n" << MESSAGE::Flush;
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
            if ( dispNeedsOwnWeight )
            {   // the dispersion's mass-weighted mean/normalizer must survive the partition merge
                quantities->disp_weight.swap( dispWeight );
                quantities->disp_velocity.swap( dispVel );
            }
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
                    if (field.velocity_dispersion && !dispNeedsOwnWeight)
                    {
                        // sigma_ij = <v_i v_j> - <v_i><v_j>; quantities->velocity is now <v>.
                        // (Under --ps-volume-weighted the dispersion has its OWN mass-weighted
                        //  mean/normalizer and is closed out in the separate pass below.)
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

        // --ps-volume-weighted, serial path: close out the MASS-weighted dispersion against its
        // own mean/normalizer (the loop above normalized 'velocity' by the VOLUME weight, so it
        // is not the right mean here). Mirrors normalizePhaseSpace's second pass.
        if ( !deferNorm && dispNeedsOwnWeight )
            for (size_t i = 0; i < totalGrid; ++i)
            {
                Real const wm = dispWeight[i];
                if ( wm <= Real(0.) ) continue;
                Real const invm = Real(1.) / wm;
                Pvector<Real,noVelComp> const vbar = dispVel[i] * invm;   // <v>_mass
                size_t c = 0;
                for (int a = 0; a < NO_DIM; ++a)
                    for (int b = a; b < NO_DIM; ++b)
                    {
                        Real s = quantities->velocity_dispersion[i][c] * invm - vbar[a]*vbar[b];
                        if (a == b and s < Real(0.)) s = Real(0.);   // variance: clamp FP-noise negatives
                        quantities->velocity_dispersion[i][c] = s;
                        ++c;
                    }
            }
    }

    // density accumulated MASS (sum of per-tetrahedron mass shares); convert to a density by dividing
    // by the grid-cell volume, then normalize by averageDensity so the written field is rho/rho_bar
    // (mean-normalized "density contrast + 1", the SAME convention as standard DTFE).
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
            // mean streams over the cell. Under psExact 'streamCount' is the raw tet-TOUCH count
            // ('.tetTouch', up to hundreds), NOT a multiplicity -- the physical one is exactMult,
            // which is what '.streams' exports. Reading streamCount here reported 110 instead of
            // 3.0 and 100% multi-stream instead of 16.67% on the pancake.
            Real const avg = psExact ? exactMult[i] : Real(streamCount[i]) * invSamples;
            if (avg > maxStreams) maxStreams = avg;
            // '> 1' to a tolerance: exactMult is a float sum, so single-stream cells sit at
            // 1.0 +/- eps (the sampled count is an exact integer multiple of invSamples).
            if (psExact ? (avg > Real(1.) + PS_STREAM_TOL)
                        : (streamCount[i] > (int)nSamplesPerCell)) multiStreamCells++;
            if (streamCount[i] > 0) coveredCells++;   // geometric coverage: any tet touches the cell
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

    // Export the per-grid-point stream count: the sampled deposit's sample-mean multiplicity,
    // or -- under --ps-exact-deposit -- the ANALYTIC cell-mean multiplicity (1/V_cell) sum V_int,
    // its exact nSub->infinity limit. Both are the same physical quantity on the same scale, so
    // single-stream cells select the same way under either deposit -- but only TO A TOLERANCE
    // (PS_STREAM_TOL / dtfelib.STREAM_TOL): this is a float, never exactly 1.
    if ( psExact )
        quantities->stream_count.swap(exactMult);   // same type and exactMult is dead below: no copy
    else
    {
        quantities->stream_count.resize(totalGrid);
        for (size_t i = 0; i < totalGrid; ++i)
            quantities->stream_count[i] = Real(streamCount[i]) * invSamples;
    }

    // --ps-exact-deposit: also export the raw integer tet-touch count ('.tetTouch'), the
    // number of tetrahedra with a nonzero intersection with the cell. Kept as a geometric
    // diagnostic (tessellation multiplicity, hundreds in resolved regions); it is NOT a
    // stream count and is not comparable to the sampled deposit's '.streams'.
    if ( psExact )
    {
        quantities->tet_touch.resize(totalGrid);
        for (size_t i = 0; i < totalGrid; ++i)
            quantities->tet_touch[i] = Real(streamCount[i]);
    }

    // Export the caustic orientation bits (0..3, exact in float); OR-merged across partitions
    // and binarized to the 0/1 fold flag once in DTFE() after the merge.
    if ( psCaustics )
    {
        quantities->caustic_bits.resize(totalGrid);
        for (size_t i = 0; i < totalGrid; ++i)
            quantities->caustic_bits[i] = Real(orientBits[i]);
    }

    // Record the sub-grid box so the caller (addFromSubgrid) can map results back into the full shared grid.
    for (int d = 0; d < NO_DIM; ++d)
    {
        quantities->ps_subOrigin[d] = subOrigin[d];
        quantities->ps_subDims[d]   = subDims[d];
    }
}

#endif // PHASE_SPACE
