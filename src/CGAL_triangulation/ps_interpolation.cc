/* Phase-space DTFE grid interpolation. Triangulation is built in Lagrangian space;
   Eulerian simplices can overlap (multi-stream), so we iterate over all cells and
   accumulate each simplex's contribution to every grid point it contains. */

#ifdef PHASE_SPACE

// PS-DTFE is validated only in 3D; the NO_DIM==2 path compiles but is untested.
#if NO_DIM==2
#warning "PS-DTFE (PHASE_SPACE) is not validated for NO_DIM==2; the 2D path is untested."
#endif

#include "triangulation_common.h"


/* Interpolates fields onto a regular grid via PS-DTFE: scatters each finite cell's
   contribution to overlapping grid points so multi-stream regions are handled.
   nSub = sub-samples per axis for volume-averaged fields (1 = unaveraged). */
void interpolateGrid_phaseSpace(DT &dt,
                                User_options &userOptions,
                                Quantities *quantities,
                                Field &field,
                                int nSub)
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

    size_t totalGrid = 1;
    for (int d = 0; d < NO_DIM; ++d) totalGrid *= nGrid[d];

    // Collect finite cell handles up front (CGAL iterators are not random-access).
    size_t noTotalCells = 0;
#if NO_DIM==2
    noTotalCells = dt.number_of_faces();
#elif NO_DIM==3
    noTotalCells = dt.number_of_finite_cells();
#endif
    std::vector<Cell_handle> cellHandles;
    cellHandles.reserve( noTotalCells );
#if NO_DIM==2
    for (DT::Finite_faces_iterator itC = dt.finite_faces_begin(); itC != dt.finite_faces_end(); ++itC)
#elif NO_DIM==3
    for (DT::Finite_cells_iterator itC = dt.finite_cells_begin(); itC != dt.finite_cells_end(); ++itC)
#endif
        cellHandles.push_back( itC );

    // psUseSubgrid: store only the Eulerian bbox of cells this partition touches, to bound peak
    // memory. Marks per axis which grid planes any kept cell overlaps (same filter as the scatter,
    // so the box covers every written cell); an axis reaching both ends stays full. off -> full grid.
    size_t subOrigin[NO_DIM], subDims[NO_DIM];
    for (int d = 0; d < NO_DIM; ++d) { subOrigin[d] = 0; subDims[d] = nGrid[d]; }
    if ( userOptions.psUseSubgrid )
    {
        std::vector<char> touched[NO_DIM];
        for (int d = 0; d < NO_DIM; ++d) touched[d].assign(nGrid[d], char(0));
        for (size_t ci = 0; ci < cellHandles.size(); ++ci)
        {
            Cell_handle cell = cellHandles[ci];
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
    std::vector<Real> massWeight;
    if ( needWeight ) massWeight.assign(totalGrid, Real(0.));

    if ( not userOptions.psSuppressGridStats )
        message << "\nPS-DTFE: Interpolating fields to grid by iterating over all Delaunay cells ...\n" << MESSAGE::Flush;

    // per-cell scatter is serial; parallelism is one level up over Lagrangian partitions
    // (DTFE.cpp). A per-cell OpenMP scatter was memory-bandwidth bound and gave no speedup.
    for (size_t ci = 0; ci < cellHandles.size(); ++ci)
    {
        Cell_handle cell = cellHandles[ci];

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

        // non-periodic hull cell density: rho = avgDensity * |det(Lag edges)|/|det(Eul edges)| (simplex factors cancel)
        Real rhoCell = Real(0.);
        if (useVolumeRatioDensity)
        {
            double Lag[NO_DIM][NO_DIM];
            for (int v = 0; v < NO_DIM; ++v)
                for (int i = 0; i < NO_DIM; ++i)
                    Lag[v][i] = double(cell->vertex(v+1)->point()[i]) - double(cell->vertex(0)->point()[i]);
            double absDetLag = std::fabs(determinant(Lag));
            rhoCell = Real( double(userOptions.averageDensity) * absDetLag / cellAbsDet );
            if (rhoCell < Real(0.)) rhoCell = Real(0.);
        }

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

        // Constant density gradient across this linear cell (from wrapped Eulerian positions).
        Real densGrad[NO_DIM];
        {
            Real dens[NO_DIM];
            for (int i = 0; i < NO_DIM; ++i)
                dens[i] = cell->vertex(i+1)->info().density() - cell->vertex(0)->info().density();
            matrixMultiplication(posMatInv, dens, densGrad);
        }

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

#if NO_DIM==2
        for (int gi = iMin[0]; gi < iMax[0]; ++gi)
        for (int gj = iMin[1]; gj < iMax[1]; ++gj)
        {
            // Wrap grid indices into [0,nGrid) for periodic boxes.
            int wgi = userOptions.periodic ? ((gi % (int)nGrid[0] + (int)nGrid[0]) % (int)nGrid[0]) : gi;
            int wgj = userOptions.periodic ? ((gj % (int)nGrid[1] + (int)nGrid[1]) % (int)nGrid[1]) : gj;
            int gridIdx[2] = {wgi, wgj};
#elif NO_DIM==3
        for (int gi = iMin[0]; gi < iMax[0]; ++gi)
        for (int gj = iMin[1]; gj < iMax[1]; ++gj)
        for (int gk = iMin[2]; gk < iMax[2]; ++gk)
        {
            // Wrap grid indices into [0,nGrid) for periodic boxes.
            int wgi = userOptions.periodic ? ((gi % (int)nGrid[0] + (int)nGrid[0]) % (int)nGrid[0]) : gi;
            int wgj = userOptions.periodic ? ((gj % (int)nGrid[1] + (int)nGrid[1]) % (int)nGrid[1]) : gj;
            int wgk = userOptions.periodic ? ((gk % (int)nGrid[2] + (int)nGrid[2]) % (int)nGrid[2]) : gk;
            int gridIdx[3] = {wgi, wgj, wgk};
#endif

            // flat index within this partition's sub-grid; inSub guard is defensive
            size_t flatIdx = 0;
            bool inSub = true;
            for (int d = 0; d < NO_DIM; ++d)
            {
                long loc = (long)gridIdx[d] - (long)subOrigin[d];
                if (loc < 0 || loc >= (long)subDims[d]) { inSub = false; break; }
                flatIdx = flatIdx * subDims[d] + (size_t)loc;
            }
            if (!inSub) continue;

            // Unwrapped raw index, used for the sample position so it stays consistent with this
            // cell's wrapped Eulerian coordinates.
#if NO_DIM==2
            int rawIdx[2] = {gi, gj};
#elif NO_DIM==3
            int rawIdx[3] = {gi, gj, gk};
#endif
            // Sub-sample the grid cell on an nSub^NO_DIM grid (cell centre when nSub==1); density and
            // stream count are divided by nSamplesPerCell afterwards to form the cell average.
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

            // Point-in-Eulerian-simplex test, relative to (wrapped) vertex 0.
            Real rel[NO_DIM];
            for (int d = 0; d < NO_DIM; ++d)
                rel[d] = gridPos[d] - eulerPos[0][d];
            // Ax stores edges as rows, so bary = (Ax^-1)^T * rel: transpose posMatInv here
            // (gradients below correctly use posMatInv directly)
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

            // stream density at this grid point (clamped >=0); doubles as the mass weight.
            // Non-periodic hull cells use the constant cell density rhoCell.
            Real densVal;
            if (useVolumeRatioDensity)
                densVal = rhoCell;
            else
            {
                densVal = cell->vertex(0)->info().density();
                for (int d = 0; d < NO_DIM; ++d)
                    densVal += rel[d] * densGrad[d];
            }
            if (densVal < Real(0.)) densVal = Real(0.);

            // Scatter this stream's contribution into the output grids.
            if (field.density) quantities->density[flatIdx] += densVal;
            if (needWeight)    massWeight[flatIdx]          += densVal;

            // Velocity moment sum(rho_s*v_s); also evaluated when dispersion is requested (it needs <v>).
            Pvector<Real,noVelComp> velVal;
            if (haveVel)
            {
                velVal = cell->vertex(0)->info().velocity();
                for (int i = 0; i < NO_DIM; ++i)
                    for (size_t j = 0; j < noVelComp; ++j)
                        velVal[j] += velGrad[i][j] * rel[i];
                quantities->velocity[flatIdx] += velVal * densVal;
            }

            // Velocity-gradient moment sum(rho_s * grad v_s) (density-weighted).
            if (field.velocity_gradient)
            {
                Pvector<Real,noGradComp> grad;
                for (size_t j = 0; j < noVelComp; ++j)
                    for (int i = 0; i < NO_DIM; ++i)
                        grad[j*NO_DIM+i] = velGrad[i][j];
                quantities->velocity_gradient[flatIdx] += grad * densVal;
            }

            // velocity dispersion: density-weighted second moment sum(rho_s v_i v_j)
            // (upper triangle). After normalization -> sigma_ij = <v_i v_j> - <v_i><v_j>.
            if (field.velocity_dispersion)
            {
                size_t c = 0;
                for (int i = 0; i < NO_DIM; ++i)
                    for (int j = i; j < NO_DIM; ++j)
                        quantities->velocity_dispersion[flatIdx][c++] += densVal * velVal[i] * velVal[j];
            }

#ifdef SCALAR
            if (field.scalar)
            {
                Pvector<Real,noScalarComp> scalarVal = cell->vertex(0)->info().myScalar();
                for (int i = 0; i < NO_DIM; ++i)
                    for (size_t j = 0; j < noScalarComp; ++j)
                        scalarVal[j] += sGrad[i][j] * rel[i];
                quantities->scalar[flatIdx] += scalarVal * densVal;
            }
            if (field.scalar_gradient)
            {
                Pvector<Real,noScalarGradComp> sgrad;
                for (size_t j = 0; j < noScalarComp; ++j)
                    for (int i = 0; i < NO_DIM; ++i)
                        sgrad[j*NO_DIM+i] = sGrad[i][j];
                quantities->scalar_gradient[flatIdx] += sgrad * densVal;
            }
#endif

            streamCount[flatIdx]++;
            }   // end grid-cell sub-sample loop
        }
    }   // end cell loop

    if ( not userOptions.psSuppressGridStats )
        message << "Done.\n" << MESSAGE::Flush;

    // average density and stream count over sub-samples (no-op when nSub==1); velocity/scalar
    // are mass-weighted means so the 1/nSamples factor cancels (not rescaled)
    Real const invSamples = Real(1.) / Real(nSamplesPerCell);
    if ( field.density )
        for (size_t i = 0; i < totalGrid; ++i)
            quantities->density[i] *= invSamples;

    // turn density-weighted moments into mass-weighted means sum(rho_s f_s)/sum(rho_s). Serial: here.
    // deferNorm: hand un-normalized moments + weight to the caller, which sums across partitions and
    // normalizes once (the only order correct for cells spanning partitions).
    if ( needWeight )
    {
        if ( deferNorm )
            quantities->mass_weight = massWeight;   // sum(rho_s); normalized later by caller
        else
            for (size_t i = 0; i < totalGrid; ++i)
                if ( massWeight[i] > Real(0.) )
                {
                    Real const inv = Real(1.) / massWeight[i];
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
        message << "PS-DTFE: Max streams at a grid point: " << maxStreams
                << ", grid points with multi-stream: " << multiStreamCells
                << " (" << (100.*multiStreamCells/totalGrid) << "\%)\n" << MESSAGE::Flush;
        message << "PS-DTFE: grid coverage: " << coveredCells << "/" << totalGrid
                << " (" << (100.*coveredCells/totalGrid) << "\%)"
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
