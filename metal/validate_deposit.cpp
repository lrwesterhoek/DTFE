//  PS-DTFE Metal deposit -- correctness validation harness.
//
//  Validates the Metal kernels against a float-exact CPU reference AND against analytic
//  expectations. Tests:
//    T1  single tetrahedron contained in one grid cell -> all mass in that cell
//    T2  tiny tetrahedron (below sub-sample spacing) -> centroid fallback, mass conserved
//    T3  periodic wrap: tet straddling the box boundary == same tet shifted by whole cells
//    T4  degenerate (flat) tetrahedron -> dropped identically by CPU and GPU
//    T5  nSub sweep (1..4): GPU total mass == CPU total mass
//    T6  determinism: two identical GPU dispatches agree to atomic-order ULP
//    T7  random stress set (normal + flat/caustic + tiny): all output grids CPU vs GPU
//    T8  Zel'dovich pancake (real multi-stream caustic flow, 6-tet Freudenthal tessellation):
//        mass conservation, full-coverage, density & stream-count profiles vs ANALYTIC
//        solution of z = q + A sin(kq), dispersion invariants (sigma_xx=sigma_yy=0 for
//        z-only flow; single-stream sigma_zz ~ 0 at nSub=1), CPU vs GPU on all fields.
//
//  Field tests run when the 'depositFields' kernel exists in the library; density-only
//  parity runs against 'depositDensity' otherwise (Phase A gate).
//
//  Build: bash metal/build_prototype.sh          Run: ./metal/validate_deposit

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

struct DepositParams {              // must match the MSL struct byte-for-byte
    float    boxLo[3];
    float    dx[3];
    int32_t  nGrid[3];
    int32_t  nSub;
    int32_t  periodic;
    int32_t  subOrigin[3];          // partition sub-grid box (full grid: origin 0, dims = nGrid)
    int32_t  subDims[3];
    uint32_t nTet;
};

static int  g_pass = 0, g_fail = 0;
static void verdict(const char* name, bool ok, const char* detail) {
    printf("  [%s] %-34s %s\n", ok ? "PASS" : "FAIL", name, detail);
    if (ok) g_pass++; else g_fail++;
}

// ------------------------------------------------------------------ CPU reference (float math,
// identical expressions to the kernel so classifications agree; only summation order differs)
static float det3(const float A[3][3]) {
    return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
         - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
         + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
}
static bool inverse3(const float A[3][3], float inv[3][3]) {
    float d = det3(A);
    if (std::fabs(d) < 1.0e-6f) return false;
    float q = 1.0f/d;
    inv[0][0]=(A[1][1]*A[2][2]-A[1][2]*A[2][1])*q; inv[1][0]=-(A[1][0]*A[2][2]-A[1][2]*A[2][0])*q; inv[2][0]=(A[1][0]*A[2][1]-A[1][1]*A[2][0])*q;
    inv[0][1]=-(A[0][1]*A[2][2]-A[0][2]*A[2][1])*q; inv[1][1]=(A[0][0]*A[2][2]-A[0][2]*A[2][0])*q; inv[2][1]=-(A[0][0]*A[2][1]-A[0][1]*A[2][0])*q;
    inv[0][2]=(A[0][1]*A[1][2]-A[0][2]*A[1][1])*q; inv[1][2]=-(A[0][0]*A[1][2]-A[0][2]*A[1][0])*q; inv[2][2]=(A[0][0]*A[1][1]-A[0][1]*A[1][0])*q;
    return true;
}
static inline int wrapIdx(int g, int n){ return ((g%n)+n)%n; }

struct CpuOut {
    std::vector<float>    mass;      // nCell
    std::vector<float>    mom;       // nCell*3
    std::vector<float>    m2;        // nCell*6 (xx,xy,xz,yy,yz,zz)
    std::vector<float>    grad;      // nCell*9 (j*3+i)
    std::vector<uint32_t> streams;   // nCell
};

static void cpuDeposit(const std::vector<float>& verts, const std::vector<float>& vels,
                       const std::vector<float>& masses, const DepositParams& P, CpuOut& o) {
    const bool hasV = !vels.empty();
    const size_t nCell = size_t(P.subDims[0])*P.subDims[1]*P.subDims[2];   // output = sub-grid sized
    o.mass.assign(nCell,0.f); o.streams.assign(nCell,0u);
    o.mom.assign(nCell*3,0.f); o.m2.assign(nCell*6,0.f); o.grad.assign(nCell*9,0.f);
    const int nSub=P.nSub; const uint32_t nSamp=uint32_t(nSub)*nSub*nSub;
    // wrapped grid cell -> sub-grid flat index; false = cell outside this partition's box
    auto subFlat=[&](int wg0,int wg1,int wg2,size_t& flat)->bool{
        int l0=wg0-P.subOrigin[0], l1=wg1-P.subOrigin[1], l2=wg2-P.subOrigin[2];
        if(l0<0||l0>=P.subDims[0]||l1<0||l1>=P.subDims[1]||l2<0||l2>=P.subDims[2]) return false;
        flat=(size_t(l0)*P.subDims[1]+l1)*P.subDims[2]+l2; return true;
    };

    for (uint32_t t=0; t<P.nTet; ++t) {
        float v[4][3], u[4][3];
        for (int k=0;k<4;++k) for (int i=0;i<3;++i) v[k][i]=verts[t*12+k*3+i];
        if (hasV) for (int k=0;k<4;++k) for (int i=0;i<3;++i) u[k][i]=vels[t*12+k*3+i];
        float m=masses[t]; if (m<=0.f) continue;

        float Ax[3][3];
        for (int e=0;e<3;++e) for (int i=0;i<3;++i) Ax[e][i]=v[e+1][i]-v[0][i];
        float d=det3(Ax);
        float avgEdge2=0.f; for(int e=0;e<3;++e){float l2=0;for(int i=0;i<3;++i)l2+=Ax[e][i]*Ax[e][i];avgEdge2+=l2;} avgEdge2/=3.f;
        if (std::fabs(d) < 1.0e-6f*avgEdge2*std::sqrt(avgEdge2)) continue;
        float posInv[3][3]; if (!inverse3(Ax,posInv)) continue;

        float vG[3][3]={{0}};
        if (hasV) {
            float dV[3][3];
            for (int e=0;e<3;++e) for (int j=0;j<3;++j) dV[e][j]=u[e+1][j]-u[0][j];
            for (int i=0;i<3;++i) for (int j=0;j<3;++j){ float s=0; for(int k=0;k<3;++k) s+=posInv[i][k]*dV[k][j]; vG[i][j]=s; }
        }

        float lo[3],hi[3]; for(int i=0;i<3;++i){lo[i]=hi[i]=v[0][i];}
        for(int k=1;k<4;++k)for(int i=0;i<3;++i){lo[i]=std::min(lo[i],v[k][i]);hi[i]=std::max(hi[i],v[k][i]);}
        int iMin[3],iMax[3];
        for(int dd=0;dd<3;++dd){
            iMin[dd]=int(std::floor((lo[dd]-P.boxLo[dd])/P.dx[dd]));
            iMax[dd]=int(std::floor((hi[dd]-P.boxLo[dd])/P.dx[dd]))+1;
            if(nSub>1){iMin[dd]-=1;iMax[dd]+=1;}
            if(!P.periodic){if(iMin[dd]<0)iMin[dd]=0;if(iMax[dd]>P.nGrid[dd])iMax[dd]=P.nGrid[dd];}
        }
        auto samplePos=[&](uint32_t s,int dd,int raw)->float{
            uint32_t rem=s; for(int k=0;k<dd;++k) rem/=uint32_t(nSub);
            float fr=(nSub==1)?0.5f:(float(int(rem%uint32_t(nSub)))+0.5f)/float(nSub);
            return P.boxLo[dd]+(float(raw)+fr)*P.dx[dd];
        };
        auto insideRel=[&](uint32_t s,int gi,int gj,int gk,float rel[3])->bool{
            int raw[3]={gi,gj,gk};
            rel[0]=samplePos(s,0,raw[0])-v[0][0]; rel[1]=samplePos(s,1,raw[1])-v[0][1]; rel[2]=samplePos(s,2,raw[2])-v[0][2];
            float sum=0.f; for(int vv=0;vv<3;++vv){float bc=0;for(int i=0;i<3;++i)bc+=posInv[i][vv]*rel[i]; if(bc<-1.0e-6f)return false; sum+=bc;}
            return sum<=1.0f+1.0e-6f;
        };
        auto depositAt=[&](size_t flat,const float rel[3],float w){
            o.mass[flat]+=w;
            if (hasV) {
                float vv[3]; for(int j=0;j<3;++j){ vv[j]=u[0][j]; for(int i=0;i<3;++i) vv[j]+=vG[i][j]*rel[i]; }
                for(int j=0;j<3;++j) o.mom[flat*3+j]+=vv[j]*w;
                size_t c=0; for(int i=0;i<3;++i) for(int j=i;j<3;++j) o.m2[flat*6+c++]+=w*vv[i]*vv[j];
                for(int j=0;j<3;++j) for(int i=0;i<3;++i) o.grad[flat*9+j*3+i]+=vG[i][j]*w;
            }
            o.streams[flat]+=1u;
        };

        // Two identical passes with the per-cell wrap + sub-grid guard resolved BEFORE sample
        // counting (matches the kernel and the production inSub guard): N counts exactly the
        // samples that get deposited.
        uint32_t N=0; float rel[3]; float share=0.f;
        for(int pass=0;pass<2;++pass){
            if(pass==1){ if(N==0) break; share=m/float(N); }
            for(int gi=iMin[0];gi<iMax[0];++gi)for(int gj=iMin[1];gj<iMax[1];++gj)for(int gk=iMin[2];gk<iMax[2];++gk){
                int wg0=P.periodic?wrapIdx(gi,P.nGrid[0]):gi, wg1=P.periodic?wrapIdx(gj,P.nGrid[1]):gj, wg2=P.periodic?wrapIdx(gk,P.nGrid[2]):gk;
                if(wg0<0||wg0>=P.nGrid[0]||wg1<0||wg1>=P.nGrid[1]||wg2<0||wg2>=P.nGrid[2])continue;
                size_t flat; if(!subFlat(wg0,wg1,wg2,flat))continue;
                for(uint32_t s=0;s<nSamp;++s) if(insideRel(s,gi,gj,gk,rel)){
                    if(pass==0) N++; else depositAt(flat,rel,share);
                }
            }
        }
        if(N==0){
            float c[3]; for(int i=0;i<3;++i)c[i]=(v[0][i]+v[1][i]+v[2][i]+v[3][i])*0.25f;
            int wg[3]; bool ok=true;
            for(int dd=0;dd<3;++dd){int cg=int(std::floor((c[dd]-P.boxLo[dd])/P.dx[dd]));wg[dd]=P.periodic?wrapIdx(cg,P.nGrid[dd]):cg;if(wg[dd]<0||wg[dd]>=P.nGrid[dd]){ok=false;break;}}
            size_t flat;
            if(ok && subFlat(wg[0],wg[1],wg[2],flat)){
                float cr[3]={c[0]-v[0][0],c[1]-v[0][1],c[2]-v[0][2]};
                depositAt(flat, cr, m);
            }
            continue;
        }
    }
}

// ------------------------------------------------------------------ GPU runner
struct Gpu {
    MTL::Device* dev=nullptr; MTL::CommandQueue* q=nullptr;
    MTL::ComputePipelineState *psoDen=nullptr, *psoFld=nullptr;

    bool init(const char* srcPath) {
        dev=MTL::CreateSystemDefaultDevice(); if(!dev) return false;
        q=dev->newCommandQueue();
        NS::Error* err=nullptr; MTL::Library* lib=nullptr;
        { std::ifstream mf("metal/ps_deposit.metallib");
          if (mf.good()) lib=dev->newLibrary(NS::URL::fileURLWithPath(NS::String::string("metal/ps_deposit.metallib",NS::UTF8StringEncoding)),&err); }
        if(!lib){
            std::ifstream f(srcPath); std::stringstream ss; ss<<f.rdbuf(); std::string src=ss.str();
            if(src.empty()) return false;
            lib=dev->newLibrary(NS::String::string(src.c_str(),NS::UTF8StringEncoding),(MTL::CompileOptions*)nullptr,&err);
        }
        if(!lib){ printf("kernel lib load fail: %s\n", err?err->localizedDescription()->utf8String():"?"); return false; }
        MTL::Function* fd=lib->newFunction(NS::String::string("depositDensity",NS::UTF8StringEncoding));
        if(fd) psoDen=dev->newComputePipelineState(fd,&err);
        // DENSITY_ONLY=1: Phase A gate -- validate the density kernel in isolation.
        if(!getenv("DENSITY_ONLY")){
            MTL::Function* ff=lib->newFunction(NS::String::string("depositFields",NS::UTF8StringEncoding));
            if(ff) psoFld=dev->newComputePipelineState(ff,&err);
        }
        return psoDen!=nullptr;
    }

    // Runs the fields kernel if available (and vels provided), else density-only.
    // Returns which mode ran: 0=density, 1=fields.
    int run(const std::vector<float>& verts, const std::vector<float>& vels,
            const std::vector<float>& masses, const DepositParams& P, CpuOut& o) {
        const size_t nCell=size_t(P.subDims[0])*P.subDims[1]*P.subDims[2];
        const bool fields = psoFld && !vels.empty();
        MTL::Buffer* bV=dev->newBuffer(verts.data(),verts.size()*sizeof(float),MTL::ResourceStorageModeShared);
        MTL::Buffer* bU=fields? dev->newBuffer(vels.data(),vels.size()*sizeof(float),MTL::ResourceStorageModeShared):nullptr;
        MTL::Buffer* bM=dev->newBuffer(masses.data(),masses.size()*sizeof(float),MTL::ResourceStorageModeShared);
        MTL::Buffer* bP=dev->newBuffer(&P,sizeof(P),MTL::ResourceStorageModeShared);
        auto zbuf=[&](size_t n){ MTL::Buffer* b=dev->newBuffer(n,MTL::ResourceStorageModeShared); memset(b->contents(),0,n); return b; };
        MTL::Buffer* bMass=zbuf(nCell*4);
        MTL::Buffer *bMom=nullptr,*bM2=nullptr,*bG=nullptr,*bS=nullptr;
        if (fields){ bMom=zbuf(nCell*12); bM2=zbuf(nCell*24); bG=zbuf(nCell*36); bS=zbuf(nCell*4); }

        MTL::CommandBuffer* cb=q->commandBuffer();
        MTL::ComputeCommandEncoder* e=cb->computeCommandEncoder();
        if (fields){
            e->setComputePipelineState(psoFld);
            e->setBuffer(bV,0,0); e->setBuffer(bU,0,1); e->setBuffer(bM,0,2);
            e->setBuffer(bMass,0,3); e->setBuffer(bMom,0,4); e->setBuffer(bM2,0,5);
            e->setBuffer(bG,0,6); e->setBuffer(bS,0,7); e->setBuffer(bP,0,8);
        } else {
            e->setComputePipelineState(psoDen);
            e->setBuffer(bV,0,0); e->setBuffer(bM,0,1); e->setBuffer(bMass,0,2); e->setBuffer(bP,0,3);
        }
        MTL::ComputePipelineState* pso = fields? psoFld : psoDen;
        NS::UInteger tg=pso->maxTotalThreadsPerThreadgroup(); if(tg>256)tg=256;
        e->dispatchThreads(MTL::Size(P.nTet,1,1),MTL::Size(tg,1,1));
        e->endEncoding(); cb->commit(); cb->waitUntilCompleted();

        o.mass.assign((float*)bMass->contents(),(float*)bMass->contents()+nCell);
        if (fields){
            o.mom.assign((float*)bMom->contents(),(float*)bMom->contents()+nCell*3);
            o.m2.assign((float*)bM2->contents(),(float*)bM2->contents()+nCell*6);
            o.grad.assign((float*)bG->contents(),(float*)bG->contents()+nCell*9);
            o.streams.assign((uint32_t*)bS->contents(),(uint32_t*)bS->contents()+nCell);
        } else { o.mom.clear(); o.m2.clear(); o.grad.clear(); o.streams.clear(); }
        return fields?1:0;
    }
};

// ------------------------------------------------------------------ helpers
static double total(const std::vector<float>& a){ double s=0; for(float x:a) s+=x; return s; }
static double meanRelDiff(const std::vector<float>& a,const std::vector<float>& b){
    double mx=0; for(float x:b) mx=std::max(mx,(double)std::fabs(x));
    if(mx==0) { double s=0; for(float x:a) s=std::max(s,(double)std::fabs(x)); return s; }
    double s=0; for(size_t i=0;i<a.size();++i) s+=std::fabs((double)a[i]-b[i]);
    return (s/a.size())/mx;
}
static char detail[512];

// ------------------------------------------------------------------ tests
static void makeParams(DepositParams& P,int n,float box,int nSub,int periodic){
    for(int i=0;i<3;++i){P.boxLo[i]=0.f;P.nGrid[i]=n;P.dx[i]=box/n;P.subOrigin[i]=0;P.subDims[i]=n;}
    P.nSub=nSub;P.periodic=periodic;P.nTet=0;
}
static std::vector<float> linVel(const std::vector<float>& verts){
    // simple linear velocity field v = (2x - y, 0.5z, -x + y) at each vertex
    std::vector<float> u(verts.size());
    for(size_t k=0;k<verts.size()/3;++k){
        float x=verts[k*3],y=verts[k*3+1],z=verts[k*3+2];
        u[k*3]=2*x-y; u[k*3+1]=0.5f*z; u[k*3+2]=-x+y;
    }
    return u;
}

static void t1_singleCell(Gpu& g){
    DepositParams P; makeParams(P,8,8.f,3,0); P.nTet=1;
    std::vector<float> v={3.1f,4.1f,5.1f, 3.9f,4.1f,5.1f, 3.1f,4.9f,5.1f, 3.1f,4.1f,5.9f};
    std::vector<float> m={2.5f};
    CpuOut c,gout; cpuDeposit(v,linVel(v),m,P,c); g.run(v,linVel(v),m,P,gout);
    size_t flat=(size_t(3)*8+4)*8+5;
    double inCellC=c.mass[flat], inCellG=gout.mass[flat];
    double totC=total(c.mass), totG=total(gout.mass);
    bool ok = std::fabs(inCellC-2.5)<1e-5 && std::fabs(inCellG-2.5)<1e-5
           && std::fabs(totC-2.5)<1e-5 && std::fabs(totG-2.5)<1e-5;
    snprintf(detail,sizeof(detail),"cell mass CPU=%.6f GPU=%.6f (expect 2.5, all in cell 3,4,5)",inCellC,inCellG);
    verdict("T1 tet inside one cell",ok,detail);
}

static void t2_centroid(Gpu& g){
    DepositParams P; makeParams(P,8,8.f,3,0); P.nTet=1;
    // Off the sample lattice: sub-samples of cell (3,4,5) sit at {3.1667,3.5,3.8333} etc., so a
    // tet of extent 0.02 at (3.62,4.57,5.43) contains NO sample point -> the N==0 centroid
    // fallback MUST run. (A previous version centred the tet exactly on the middle sub-sample,
    // which counted as inside and silently took the normal deposit path instead.)
    float cx=3.62f,cy=4.57f,cz=5.43f,e=0.02f;
    std::vector<float> v={cx,cy,cz, cx+e,cy,cz, cx,cy+e,cz, cx,cy,cz+e};
    std::vector<float> m={1.7f};
    CpuOut c,gout; cpuDeposit(v,linVel(v),m,P,c); int mode=g.run(v,linVel(v),m,P,gout);
    size_t flat=(size_t(3)*8+4)*8+5;
    bool ok = std::fabs(c.mass[flat]-1.7)<1e-6 && std::fabs(gout.mass[flat]-1.7)<1e-6
           && std::fabs(total(c.mass)-1.7)<1e-6 && std::fabs(total(gout.mass)-1.7)<1e-6
           && c.streams[flat]==1u;                        // exactly ONE deposit => fallback path ran
    if(mode==1) ok = ok && gout.streams[flat]==1u;
    snprintf(detail,sizeof(detail),"centroid cell mass CPU=%.6f GPU=%.6f streams=%u (expect 1.7 via 1 fallback deposit)",
             (double)c.mass[flat],(double)gout.mass[flat],c.streams[flat]);
    verdict("T2 tiny tet centroid fallback",ok,detail);
}

static void t3_periodicWrap(Gpu& g){
    DepositParams P; makeParams(P,8,8.f,3,1);
    // tet straddling z boundary vs same tet shifted -4 cells (=4.0, an exact multiple of dx)
    std::vector<float> vA={4.1f,4.1f,7.7f, 4.9f,4.1f,7.7f, 4.1f,4.9f,7.7f, 4.1f,4.1f,8.5f};
    std::vector<float> vB=vA; for(int k=0;k<4;++k) vB[k*3+2]-=4.f;
    std::vector<float> m={3.3f}; P.nTet=1;
    CpuOut ga,gb; g.run(vA,{},m,P,ga); g.run(vB,{},m,P,gb);
    double totA=total(ga.mass), totB=total(gb.mass);
    // compare A at z with B at z-4 (mod 8)
    double maxd=0;
    for(int i=0;i<8;++i)for(int j=0;j<8;++j)for(int k=0;k<8;++k){
        size_t fa=(size_t(i)*8+j)*8+k, fb=(size_t(i)*8+j)*8+((k+4)%8);
        maxd=std::max(maxd,std::fabs((double)ga.mass[fa]-gb.mass[fb]));
    }
    bool ok = std::fabs(totA-3.3)<1e-5 && std::fabs(totB-3.3)<1e-5 && maxd<1e-3*3.3;
    snprintf(detail,sizeof(detail),"totals %.6f/%.6f (expect 3.3), max cell mismatch after shift %.2e",totA,totB,maxd);
    verdict("T3 periodic boundary wrap",ok,detail);
}

static void t4_degenerate(Gpu& g){
    DepositParams P; makeParams(P,8,8.f,3,1); P.nTet=1;
    std::vector<float> v={2.f,2.f,4.f, 3.f,2.f,4.f, 2.f,3.f,4.f, 3.f,3.f,4.f};  // coplanar
    std::vector<float> m={9.9f};
    CpuOut c,gout; cpuDeposit(v,{},m,P,c); g.run(v,{},m,P,gout);
    bool ok = total(c.mass)==0.0 && total(gout.mass)==0.0;
    snprintf(detail,sizeof(detail),"deposited CPU=%.3e GPU=%.3e (expect 0: dropped)",total(c.mass),total(gout.mass));
    verdict("T4 degenerate tet dropped",ok,detail);
}

static void makeStress(uint32_t nTet,float box,float dxc,std::vector<float>& verts,std::vector<float>& vels,std::vector<float>& masses,
                       bool wellConditioned=false){
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> U(0.f,box),Um(0.5f,1.5f),Uc(0.f,1.f);
    std::normal_distribution<float> Nrm(0.f,1.f),Nv(0.f,300.f);
    verts.resize(nTet*12); vels.resize(nTet*12); masses.resize(nTet);
    for(uint32_t t=0;t<nTet;++t){
        float b[3]={U(rng),U(rng),U(rng)}; float r=Uc(rng),scale,thin;
        if(wellConditioned){scale=dxc*(1.f+2.f*Uc(rng));thin=1.f;}   // no thin/tiny classes
        else if(r<0.60f){scale=dxc*(1.f+2.f*Uc(rng));thin=1.f;}
        else if(r<0.85f){scale=dxc*(1.f+3.f*Uc(rng));thin=0.02f;}
        else {scale=dxc*0.3f*Uc(rng);thin=1.f;}
        for(;;){
            for(int k=0;k<4;++k){
                verts[t*12+k*3+0]=b[0]+Nrm(rng)*scale;
                verts[t*12+k*3+1]=b[1]+Nrm(rng)*scale;
                verts[t*12+k*3+2]=b[2]+Nrm(rng)*scale*thin;
                for(int i=0;i<3;++i) vels[t*12+k*3+i]=Nv(rng);
            }
            if(!wellConditioned) break;
            // wellConditioned: reject near-coplanar draws (random Gaussian tets have a det->0
            // tail whose float conditioning would dominate any analytic plumbing check)
            float Ax[3][3];
            for(int e=0;e<3;++e)for(int i=0;i<3;++i) Ax[e][i]=verts[t*12+(e+1)*3+i]-verts[t*12+i];
            float d=det3(Ax);
            float a2=0; for(int e=0;e<3;++e){float l2=0;for(int i=0;i<3;++i)l2+=Ax[e][i]*Ax[e][i];a2+=l2;} a2/=3.f;
            if(std::fabs(d)>=0.05f*a2*std::sqrt(a2)) break;
        }
        masses[t]=Um(rng);
    }
}

static void t5_nSubSweep(Gpu& g){
    std::vector<float> v,u,m; makeStress(2000,10.f,10.f/32.f,v,u,m);
    bool ok=true; std::string ds;
    for(int ns=1;ns<=4;++ns){
        DepositParams P; makeParams(P,32,10.f,ns,1); P.nTet=2000;
        CpuOut c,go; cpuDeposit(v,{},m,P,c); g.run(v,{},m,P,go);
        double rc=total(go.mass)/total(c.mass);
        char b[64]; snprintf(b,sizeof(b),"nSub%d:%.7f ",ns,rc); ds+=b;
        if(std::fabs(rc-1.0)>1e-4) ok=false;
    }
    verdict("T5 nSub sweep mass GPU/CPU",ok,ds.c_str());
}

static void t6_determinism(Gpu& g){
    std::vector<float> v,u,m; makeStress(20000,10.f,10.f/32.f,v,u,m);
    DepositParams P; makeParams(P,32,10.f,3,1); P.nTet=20000;
    CpuOut a,b; g.run(v,{},m,P,a); g.run(v,{},m,P,b);
    double mx=0,pk=0; for(size_t i=0;i<a.mass.size();++i){mx=std::max(mx,std::fabs((double)a.mass[i]-b.mass[i]));pk=std::max(pk,(double)a.mass[i]);}
    bool ok = mx/pk < 1e-5;
    snprintf(detail,sizeof(detail),"max run-to-run diff %.2e of peak (atomic-order ULP)",mx/pk);
    verdict("T6 determinism (2 dispatches)",ok,detail);
}

static void t7_stressFields(Gpu& g){
    std::vector<float> v,u,m; makeStress(20000,10.f,10.f/32.f,v,u,m);
    DepositParams P; makeParams(P,32,10.f,3,1); P.nTet=20000;
    CpuOut c,go; cpuDeposit(v,u,m,P,c);
    int mode=g.run(v,u,m,P,go);
    double dm=meanRelDiff(go.mass,c.mass);
    bool ok = std::fabs(total(go.mass)/total(c.mass)-1.0)<1e-4 && dm<1e-4;
    if(mode==0){
        snprintf(detail,sizeof(detail),"mass mean-rel %.2e (fields kernel NOT present; density-only parity)",dm);
        verdict("T7 stress set CPU vs GPU",ok,detail);
        return;
    }
    double dmm=meanRelDiff(go.mom,c.mom), dm2=meanRelDiff(go.m2,c.m2), dg=meanRelDiff(go.grad,c.grad);
    size_t smis=0; uint32_t smax=0;
    for(size_t i=0;i<c.streams.size();++i){ uint32_t d=c.streams[i]>go.streams[i]?c.streams[i]-go.streams[i]:go.streams[i]-c.streams[i]; if(d){smis++; smax=std::max(smax,d);} }
    double smisf=double(smis)/c.streams.size();
    ok = ok && dmm<1e-3 && dm2<1e-3 && dg<1e-3 && smisf<5e-3;
    snprintf(detail,sizeof(detail),"mass %.1e mom %.1e m2 %.1e grad %.1e | streams mismatch %.3f%% (max %u)",
             dm,dmm,dm2,dg,100*smisf,smax);
    verdict("T7 stress set all fields CPU vs GPU",ok,detail);
}

// ---- T8: Zel'dovich pancake with analytic reference --------------------------------
static void t8_zeldovich(Gpu& g,int nSub){
    // N=45 gives dq / (dx/nSub) = 4.267 (non-integer) and per-axis phase offsets keep sub-sample
    // points off the (displacement-invariant) x=y Freudenthal faces: a commensurate lattice makes
    // samples sit EXACTLY on shared tet faces (counted in both neighbours by the +-1e-6 tolerance)
    // and repeats one quantization pattern in every tet (moire in the void profile) -- test-setup
    // artifacts of a perfectly regular grid, absent for irregular/displaced real data.
    const int   N=45, NG=64;
    const float L=100.f, dq=L/N, k=2.f*float(M_PI)/L, A=1.8f/k;   // A*k=1.8 -> 3-stream pancake
    const float rhoBar=1.f, mTet=dq*dq*dq/6.f;
    const float off[3]={0.171f,0.313f,0.457f};   // per-axis Lagrangian phase offsets (units of dq)
    // Small linear shear z += alpha*x + beta*y: an affine map (tets stay conforming, Jacobian
    // unchanged) that gives every (x,y) column a different z-phase relative to the grid, so tet
    // quantization noise decorrelates across a slab instead of adding coherently. The exact
    // expected profile becomes the 1D PL pushforward convolved with two box kernels of widths
    // alpha*L and beta*L (the shift alpha*x+beta*y is uniform per axis). Kernel widths are integer
    // multiples of the fine reference bin so the convolution below is exact.
    const int FSUB=64;                                    // fine sub-bins per profile bin
    const int wA=152, wB=88;                              // box widths in fine bins (of L/(NG*FSUB))
    const float alpha=float(wA)/(NG*FSUB), beta=float(wB)/(NG*FSUB);
    // Freudenthal 6-tet split of each Lagrangian cube; vertices mapped by x=q+A sin(k qz) ez.
    static const int perms[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    std::vector<float> verts,vels,masses;
    verts.reserve(size_t(N)*N*N*6*12); vels.reserve(size_t(N)*N*N*6*12); masses.reserve(size_t(N)*N*N*6);
    auto emit=[&](float qx,float qy,float qz){
        float disp=A*std::sin(k*qz);
        verts.push_back(qx); verts.push_back(qy); verts.push_back(qz+disp+alpha*qx+beta*qy);
        vels.push_back(0.f); vels.push_back(0.f); vels.push_back(100.f*disp);
    };
    for(int i=0;i<N;++i)for(int j=0;j<N;++j)for(int kk=0;kk<N;++kk)
        for(int p=0;p<6;++p){
            float w[3]={(i+off[0])*dq,(j+off[1])*dq,(kk+off[2])*dq};
            emit(w[0],w[1],w[2]);
            for(int step=0;step<3;++step){ w[perms[p][step]]+=dq; emit(w[0],w[1],w[2]); }
            masses.push_back(mTet);
        }
    DepositParams P; makeParams(P,NG,L,nSub,1); P.nTet=uint32_t(masses.size());
    const size_t nCell=size_t(NG)*NG*NG; const float cellVol=(L/NG)*(L/NG)*(L/NG);
    const uint32_t nSamp=uint32_t(nSub)*nSub*nSub;

    CpuOut c,go; cpuDeposit(verts,vels,masses,P,c);
    int mode=g.run(verts,vels,masses,P,go);
    double expMass=double(P.nTet)*mTet;   // == rhoBar*L^3
    double cTot=total(c.mass), gTot=total(go.mass);

    // EXACT expected profiles of the tet ensemble: for a z-only displacement, the Freudenthal
    // interpolant depends only on q_z, so the deposit's continuum limit is the pushforward of the
    // 1D PIECEWISE-LINEAR map z_i = q_i + A sin(k q_i) (nodes = the tet grid). Each q-interval
    // spreads mass rhoBar*dq uniformly over [z_i, z_i+1]; stream presence adds its bin-coverage
    // fraction. This removes tessellation error from the comparison and tests ONLY the deposit.
    std::vector<double> rhoAn(NG,0.0), strAn(NG,0.0);
    {
        // 1) PL pushforward at fine resolution (FSUB sub-bins per profile bin)
        const int NF=NG*FSUB; const double dzF=double(L)/NF;
        std::vector<double> rhoF(NF,0.0), strF(NF,0.0);
        auto addSpan=[&](double z0,double z1,double wRho){
            if(z1<=z0) return;
            int b0=int(std::floor(z0/dzF)), b1=int(std::floor((z1-1e-12)/dzF));
            for(int b=b0;b<=b1;++b){
                double lo=std::max(z0,b*dzF), hi=std::min(z1,(b+1)*dzF);
                if(hi<=lo) continue;
                int bw=((b%NF)+NF)%NF;
                rhoF[bw]+=wRho*(hi-lo);
                strF[bw]+=(hi-lo)/dzF;
            }
        };
        for(int i=0;i<N;++i){
            double q0=(i+off[2])*double(dq), q1=(i+1+off[2])*double(dq);
            double z0=q0+A*std::sin(k*q0), z1=q1+A*std::sin(k*q1);
            double zl=std::min(z0,z1), zh=std::max(z0,z1), len=zh-zl;
            double offw=std::floor(zl/L)*L; zl-=offw; zh-=offw;
            if(len<1e-9){ int b=((int(zl/dzF))%NF+NF)%NF; rhoF[b]+=rhoBar*double(dq); continue; }
            double wRho=rhoBar*double(dq)/len;
            if(zh<=L) addSpan(zl,zh,wRho);
            else { addSpan(zl,L,wRho); addSpan(0.0,zh-L,wRho); }
        }
        // 2) shear = two periodic box convolutions (widths wA,wB fine bins), plus the constant
        //    shift alpha*x0+beta*y0 from the Lagrangian phase offsets (x starts at off*dq).
        auto boxConv=[&](std::vector<double>& a,int w){
            if(w<=1) return;
            std::vector<double> cum(NF+1,0.0);
            std::vector<double> outv(NF);
            for(int i=0;i<NF;++i) cum[i+1]=cum[i]+a[i];
            auto rangeSum=[&](int s,int e)->double{   // sum a[(s..e-1) mod NF]
                double tot=0; int len=e-s;
                s=((s%NF)+NF)%NF;
                while(len>0){ int take=std::min(len,NF-s); tot+=cum[s+take]-cum[s]; s=0; len-=take; }
                return tot;
            };
            for(int i=0;i<NF;++i) outv[i]=rangeSum(i-w+1,i+1)/w;
            a.swap(outv);
        };
        int shift=int(std::llround((alpha*(off[0]*dq)+beta*(off[1]*dq))/dzF));
        boxConv(rhoF,wA); boxConv(strF,wB==0?1:wB);
        boxConv(rhoF,wB==0?1:wB); boxConv(strF,wA);
        // 3) aggregate to profile bins (apply the small constant shift). rhoF holds MASS per fine
        //    bin -> mean slab density = (sum of fine-bin masses)/dz; strF holds coverage fraction
        //    per fine bin -> bin coverage = mean over its FSUB fine bins.
        const double dz=double(L)/NG;
        for(int b=0;b<NG;++b){
            double r=0,s=0;
            for(int f=0;f<FSUB;++f){ int idx=((b*FSUB+f-shift)%NF+NF)%NF; r+=rhoF[idx]; s+=strF[idx]; }
            rhoAn[b]=r/dz;
            strAn[b]=s/FSUB;
        }
    }

    // deposited slab profiles (use GPU grids when fields ran, else CPU for fields)
    const CpuOut& F = (mode==1)? go : c;
    std::vector<double> rhoDep(NG,0.0), strDep(NG,0.0);
    for(int i=0;i<NG;++i)for(int j=0;j<NG;++j)for(int kk=0;kk<NG;++kk){
        size_t f=(size_t(i)*NG+j)*NG+kk;
        rhoDep[kk]+=F.mass[f]/cellVol; strDep[kk]+=double(F.streams.empty()?0.f:(double)F.streams[f])/nSamp;
    }
    for(int b=0;b<NG;++b){ rhoDep[b]/=double(NG)*NG; strDep[b]/=double(NG)*NG; }

    // compare deposited slab profiles against the EXACT piecewise-linear expectation. Bins with
    // extreme expected density (>10 rhoBar, the caustic bins) are excluded: there a tet's z-extent
    // is far below the sample spacing, so per-bin sampling quantization dominates (mass still lands
    // in the right place to within a cell -- conservation is checked separately).
    double rhoErr=0, strErr=0; int used=0;
    if(getenv("DUMP_PROFILES")){
        printf("   bin      rhoAn    rhoDep     strAn    strDep\n");
        for(int b=0;b<NG;++b) printf("   %3d   %8.4f  %8.4f  %8.4f  %8.4f\n",b,rhoAn[b],rhoDep[b],strAn[b],strDep[b]);
    }
    for(int b=0;b<NG;++b){
        if(rhoAn[b]>10.0*rhoBar) continue;
        rhoErr=std::max(rhoErr,std::fabs(rhoDep[b]-rhoAn[b])/rhoAn[b]);
        strErr=std::max(strErr,std::fabs(strDep[b]-strAn[b])/strAn[b]);
        used++;
    }
    char tag[32]; snprintf(tag,sizeof(tag),"(nSub=%d)",nSub);

    snprintf(detail,sizeof(detail),"total CPU %.6g GPU %.6g expect %.6g | GPU/CPU-1 = %.1e",
             cTot,gTot,expMass,gTot/cTot-1.0);
    verdict((std::string("T8 mass conservation ")+tag).c_str(),
            std::fabs(cTot/expMass-1.0)<1e-3 && std::fabs(gTot/expMass-1.0)<1e-3, detail);

    double minMass=1e30; size_t zeroMass=0;
    for(size_t f=0;f<nCell;++f){ minMass=std::min(minMass,(double)F.mass[f]); if(F.mass[f]<=0.f) zeroMass++; }
    snprintf(detail,sizeof(detail),"min density %.3f rhoBar (PL void floor ~0.36), zero-mass cells %zu",
             minMass/cellVol/rhoBar,zeroMass);
    // nSub=1 point-samples one position per cell: per-cell noise is expected (mass still conserved);
    // require full coverage only. nSub>=3 must also stay near the physical void floor: 0.15 rhoBar
    // is ~4 sigma below the 0.36 rhoBar PL floor at the ~14% per-cell sampling noise (262k cells).
    verdict((std::string("T8 full coverage ")+tag).c_str(),
            zeroMass==0 && (nSub==1 || minMass/cellVol>0.15*rhoBar), detail);

    if (nSub>=3) {
        snprintf(detail,sizeof(detail),"max rel err vs exact PL profile: density %.2f%% streams %.2f%% (%d bins)",
                 100*rhoErr,100*strErr,used);
        verdict((std::string("T8 exact-PL profiles ")+tag).c_str(), rhoErr<0.05 && (mode==0||strErr<0.05), detail);
    } else {
        // informational at nSub=1: single-sample-per-cell quantization dominates by construction
        printf("  [info] T8 profiles %s               density %.1f%% streams %.1f%% vs exact PL (sampling quantization, not a deposit error)\n",
               tag,100*rhoErr,100*strErr);
    }

    if(mode==1){
        // normalize and check dispersion invariants + CPU/GPU agreement
        auto normalize=[&](const CpuOut& X,std::vector<float>& sig,std::vector<float>& vbar){
            sig.assign(nCell*6,0.f); vbar.assign(nCell*3,0.f);
            for(size_t f=0;f<nCell;++f){
                float W=X.mass[f]; if(W<=0.f) continue;
                for(int j=0;j<3;++j) vbar[f*3+j]=X.mom[f*3+j]/W;
                size_t cix=0;
                for(int i=0;i<3;++i)for(int j=i;j<3;++j){
                    float s=X.m2[f*6+cix]/W - vbar[f*3+i]*vbar[f*3+j];
                    if(i==j&&s<0.f)s=0.f; sig[f*6+cix]=s; ++cix; }
            }};
        std::vector<float> sigC,vbC,sigG,vbG; normalize(c,sigC,vbC); normalize(go,sigG,vbG);
        double maxOff=0,maxZZ=0;
        for(size_t f=0;f<nCell;++f){
            maxOff=std::max({maxOff,(double)std::fabs(sigG[f*6+0]),(double)std::fabs(sigG[f*6+1]),(double)std::fabs(sigG[f*6+2]),
                                    (double)std::fabs(sigG[f*6+3]),(double)std::fabs(sigG[f*6+4])});
            maxZZ=std::max(maxZZ,(double)sigG[f*6+5]);
        }
        snprintf(detail,sizeof(detail),"max |sigma_xx..yz| %.2e vs max sigma_zz %.4g (z-only flow => must be ~0)",maxOff,maxZZ);
        verdict((std::string("T8 tensor invariants ")+tag).c_str(), maxOff<1e-4*maxZZ, detail);

        if(nSub==1){
            double ssMax=0; size_t nss=0;
            for(size_t f=0;f<nCell;++f) if(go.streams[f]==1u){ nss++; ssMax=std::max(ssMax,(double)sigG[f*6+5]); }
            snprintf(detail,sizeof(detail),"%zu single-stream cells, max sigma_zz %.3f (km/s)^2 (expect ~0)",nss,ssMax);
            verdict("T8 single-stream sigma_zz=0 (nSub=1)", ssMax<5.0, detail);
        }
        double dS=meanRelDiff(sigG,sigC), dV=meanRelDiff(vbG,vbC);
        snprintf(detail,sizeof(detail),"normalized sigma mean-rel %.2e, vbar mean-rel %.2e",dS,dV);
        verdict((std::string("T8 fields CPU vs GPU ")+tag).c_str(), dS<1e-3&&dV<1e-3, detail);
    }
}

// T9: analytic velocity-field checks with real power over the velocity/moment plumbing (the
// single-stream sigma identity in T8 holds for ANY deposited velocity, so it cannot catch a wrong
// velVal). (a) CONSTANT field v=v0: every covered cell must have vbar == v0 (exact linear-interp
// reproduction), sigma == 0, gradient moment == 0. (b) LINEAR field v = G x: every tet's velocity
// gradient is exactly G, so grad/W == G in every covered cell regardless of the sample layout.
static void t9_analyticVelocity(Gpu& g){
    if(!g.psoFld){ printf("  [info] T9 skipped (fields kernel not present)\n"); return; }
    std::vector<float> v,uD,m; makeStress(20000,10.f,10.f/32.f,v,uD,m);
    DepositParams P; makeParams(P,32,10.f,3,1); P.nTet=20000;
    const size_t nCell=size_t(32)*32*32;

    // (a) constant velocity
    const float v0[3]={137.5f,-42.25f,981.f};
    std::vector<float> uC(v.size());
    for(size_t kk=0;kk<v.size()/3;++kk) for(int j=0;j<3;++j) uC[kk*3+j]=v0[j];
    CpuOut go; g.run(v,uC,m,P,go);
    double vErr=0,sErr=0,gErr=0;
    for(size_t f=0;f<nCell;++f){
        float W=go.mass[f]; if(W<=0.f) continue;
        for(int j=0;j<3;++j) vErr=std::max(vErr,(double)std::fabs(go.mom[f*3+j]/W-v0[j])/std::fabs((double)v0[j]));
        size_t c=0;
        for(int i=0;i<3;++i)for(int j=i;j<3;++j){
            double s=go.m2[f*6+c]/W - double(go.mom[f*3+i]/W)*double(go.mom[f*3+j]/W);
            sErr=std::max(sErr,std::fabs(s)/std::fabs(double(v0[i])*double(v0[j]))); ++c; }
        for(int q=0;q<9;++q) gErr=std::max(gErr,(double)std::fabs(go.grad[f*9+q]/W));
    }
    snprintf(detail,sizeof(detail),"const field: max|vbar-v0|/|v0| %.1e, max|sigma|/|v_i v_j| %.1e, max|grad| %.1e",vErr,sErr,gErr);
    verdict("T9a constant velocity analytic",vErr<1e-5&&sErr<1e-4&&gErr<1e-3,detail);

    // (b) linear velocity v = G x (row-major G[i][j] = dv_j/dx_i as the kernel stores it).
    // Well-conditioned tets only: every tet's velGrad is exactly G up to cond(Ax)*eps float
    // rounding, so grad/W == G in every covered cell. (Near-degenerate tets amplify the float
    // inverse by their condition number -- identically on CPU, covered by T7 parity -- and would
    // turn this analytic check into a conditioning test instead of a plumbing test.)
    std::vector<float> vW,uW,mW; makeStress(20000,10.f,10.f/32.f,vW,uW,mW,true);
    const float G[3][3]={{2.f,0.5f,-1.f},{-1.f,0.25f,1.f},{0.5f,-0.75f,3.f}};
    std::vector<float> uL(vW.size());
    for(size_t kk=0;kk<vW.size()/3;++kk)
        for(int j=0;j<3;++j){ float s=0; for(int i=0;i<3;++i) s+=G[i][j]*vW[kk*3+i]; uL[kk*3+j]=s; }
    CpuOut gl; g.run(vW,uL,mW,P,gl);
    double lErr=0; double gnorm=0; for(int i=0;i<3;++i)for(int j=0;j<3;++j) gnorm=std::max(gnorm,(double)std::fabs(G[i][j]));
    for(size_t f=0;f<nCell;++f){
        float W=gl.mass[f]; if(W<=0.f) continue;
        for(int j=0;j<3;++j)for(int i=0;i<3;++i)
            lErr=std::max(lErr,std::fabs(gl.grad[f*9+j*3+i]/W-G[i][j])/gnorm);
    }
    snprintf(detail,sizeof(detail),"linear field v=Gx (well-conditioned tets): max|grad/W - G|/|G| %.1e",lErr);
    verdict("T9b linear velocity gradient analytic",lErr<1e-4,detail);
}

// T10: partition sub-grid support. Tets confined to the interior deposit identically whether the
// output is the full grid or a cropped sub-grid box covering them (production stores only the
// Eulerian bbox each Lagrangian partition touches). Streams must match exactly; masses to atomic-
// order ULP; and the cropped GPU run must match the cropped CPU reference.
static void t10_subgrid(Gpu& g){
    if(!g.psoFld){ printf("  [info] T10 skipped (fields kernel not present)\n"); return; }
    std::vector<float> v0,u0,m0; makeStress(20000,10.f,10.f/32.f,v0,u0,m0,true);
    std::vector<float> v,u,m;      // keep tets fully inside [2,8] so the crop covers all touches
    for(size_t t=0;t<m0.size();++t){
        bool in=true;
        for(int k=0;k<12;++k){ float c=v0[t*12+k]; if(c<2.f||c>8.f){in=false;break;} }
        if(!in) continue;
        for(int k=0;k<12;++k){ v.push_back(v0[t*12+k]); u.push_back(u0[t*12+k]); }
        m.push_back(m0[t]);
    }
    DepositParams PF; makeParams(PF,32,10.f,3,1); PF.nTet=uint32_t(m.size());
    CpuOut full; g.run(v,u,m,PF,full);
    DepositParams PS=PF;
    for(int i=0;i<3;++i){ PS.subOrigin[i]=4; PS.subDims[i]=24; }   // crop = cells [4,28) = coords [1.25,8.75]
    CpuOut sub; g.run(v,u,m,PS,sub);
    CpuOut csub; cpuDeposit(v,u,m,PS,csub);

    double maxd=0, peak=0, totCrop=0; size_t smis=0;
    for(int a=0;a<24;++a)for(int b=0;b<24;++b)for(int c=0;c<24;++c){
        size_t fF=((size_t)(a+4)*32+(b+4))*32+(c+4);
        size_t fS=((size_t)a*24+b)*24+c;
        maxd=std::max(maxd,std::fabs((double)full.mass[fF]-sub.mass[fS]));
        peak=std::max(peak,(double)sub.mass[fS]);
        if(full.streams[fF]!=sub.streams[fS]) smis++;
        totCrop+=full.mass[fF];
    }
    double totFull=total(full.mass), totSub=total(sub.mass);
    double cpuPar=meanRelDiff(sub.mass,csub.mass);
    bool ok = smis==0 && maxd<1e-5*peak
           && std::fabs(totCrop-totFull)<1e-6*totFull        // nothing deposited outside the crop
           && std::fabs(totSub-totFull)<1e-4*totFull
           && cpuPar<1e-4;
    snprintf(detail,sizeof(detail),"%u tets: stream mismatches %zu, max mass diff %.1e of peak, CPU parity %.1e",
             PF.nTet,smis,maxd/peak,cpuPar);
    verdict("T10 partition sub-grid equivalence",ok,detail);
}

// BENCH=1: time the all-fields deposit, CPU (1 core) vs GPU, on a 1M-tet stress set.
static void bench(Gpu& g){
    if(!g.psoFld){ printf("  [bench] fields kernel missing, skipped\n"); return; }
    std::vector<float> v,u,m; makeStress(1000000,10.f,10.f/32.f,v,u,m);
    DepositParams P; makeParams(P,64,10.f,3,1); P.nTet=1000000;
    CpuOut c,go;
    auto t0=std::chrono::high_resolution_clock::now();
    cpuDeposit(v,u,m,P,c);
    auto t1=std::chrono::high_resolution_clock::now();
    g.run(v,u,m,P,go);   // warm-up (pipeline + buffer setup)
    auto t2=std::chrono::high_resolution_clock::now();
    g.run(v,u,m,P,go);
    auto t3=std::chrono::high_resolution_clock::now();
    double cpuMs=std::chrono::duration<double,std::milli>(t1-t0).count();
    double gpuMs=std::chrono::duration<double,std::milli>(t3-t2).count();
    printf("  [bench] all-fields deposit, 1M tets, grid 64^3, nSub=3: CPU(1 core) %.0f ms, GPU %.0f ms, speedup %.1fx\n",
           cpuMs,gpuMs,cpuMs/gpuMs);
}

int main(){
    Gpu g;
    if(!g.init("metal/ps_deposit.metal")){ printf("GPU init failed\n"); return 1; }
    printf("device: %s | kernels: depositDensity=%s depositFields=%s\n",
           g.dev->name()->utf8String(), g.psoDen?"yes":"NO", g.psoFld?"yes":"MISSING (density-only gate)");
    printf("------------------------------------------------------------------------\n");
    t1_singleCell(g);
    t2_centroid(g);
    t3_periodicWrap(g);
    t4_degenerate(g);
    t5_nSubSweep(g);
    t6_determinism(g);
    t7_stressFields(g);
    t8_zeldovich(g,3);
    t8_zeldovich(g,1);
    t9_analyticVelocity(g);
    t10_subgrid(g);
    if(getenv("BENCH")) bench(g);
    printf("------------------------------------------------------------------------\n");
    printf("SUMMARY: %d passed, %d failed\n",g_pass,g_fail);
    return g_fail?1:0;
}
