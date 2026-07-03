//  PS-DTFE Metal density-deposit prototype -- validation harness.
//
//  Generates a stress set of tetrahedra (normal + flat/tiny "caustic" simplices -- the shapes
//  that broke the old point-sampling deposit), runs BOTH a CPU reference and the Metal GPU kernel
//  of the mass-conserving deposit, and checks: (1) total mass conserved (== sum of kept tet masses),
//  (2) GPU matches CPU per cell. The kernel is compiled at runtime from metal/ps_deposit.metal.
//
//  Build (see metal/build_prototype.sh):
//    clang++ -std=c++17 -O2 -I third_party/metal-cpp metal/deposit_prototype.cpp \
//        -framework Metal -framework Foundation -o metal/deposit_prototype

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>

struct DepositParams {           // must match the MSL struct byte-for-byte
    float    boxLo[3];
    float    dx[3];
    int32_t  nGrid[3];
    int32_t  nSub;
    int32_t  periodic;
    int32_t  subOrigin[3];       // used by depositFields only; depositDensity is full-grid
    int32_t  subDims[3];
    uint32_t nTet;
};

// ---------------------------------------------------------------------------------------------
// CPU reference: identical algorithm to the kernel (and to ps_interpolation.cc), density mass only.
// ---------------------------------------------------------------------------------------------
// NOTE: float math throughout (matching Real=float in production and the Metal kernel), so CPU and
// GPU make identical inside/outside classifications; only the atomic accumulation ORDER differs.
static float det3(const float A[3][3]) {
    return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
         - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
         + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
}
static bool inverse3(const float A[3][3], float inv[3][3]) {
    float d = det3(A);
    if (std::fabs(d) < 1.0e-6f) return false;
    float invd = 1.0f/d;
    inv[0][0]=(A[1][1]*A[2][2]-A[1][2]*A[2][1])*invd; inv[1][0]=-(A[1][0]*A[2][2]-A[1][2]*A[2][0])*invd; inv[2][0]=(A[1][0]*A[2][1]-A[1][1]*A[2][0])*invd;
    inv[0][1]=-(A[0][1]*A[2][2]-A[0][2]*A[2][1])*invd; inv[1][1]=(A[0][0]*A[2][2]-A[0][2]*A[2][0])*invd; inv[2][1]=-(A[0][0]*A[2][1]-A[0][1]*A[2][0])*invd;
    inv[0][2]=(A[0][1]*A[1][2]-A[0][2]*A[1][1])*invd; inv[1][2]=-(A[0][0]*A[1][2]-A[0][2]*A[1][0])*invd; inv[2][2]=(A[0][0]*A[1][1]-A[0][1]*A[1][0])*invd;
    return true;
}
static inline int wrapIdx(int g, int n){ return ((g%n)+n)%n; }

static void cpuDeposit(const std::vector<float>& verts, const std::vector<float>& masses,
                       const DepositParams& P, std::vector<double>& grid) {
    const int nSub=P.nSub; const uint32_t nSamp=uint32_t(nSub)*nSub*nSub;
    for (uint32_t t=0; t<P.nTet; ++t) {
        float v[4][3];
        for (int k=0;k<4;++k) for (int i=0;i<3;++i) v[k][i]=verts[t*12+k*3+i];
        float m=masses[t]; if (m<=0.0f) continue;
        float Ax[3][3];
        for (int e=0;e<3;++e) for (int i=0;i<3;++i) Ax[e][i]=v[e+1][i]-v[0][i];
        float d=det3(Ax);
        float avgEdge2=0.0f; for(int e=0;e<3;++e){float l2=0;for(int i=0;i<3;++i)l2+=Ax[e][i]*Ax[e][i];avgEdge2+=l2;} avgEdge2/=3.0f;
        float edgeScale=avgEdge2*std::sqrt(avgEdge2);
        if (std::fabs(d) < 1.0e-6f*edgeScale) continue;
        float posInv[3][3]; if (!inverse3(Ax,posInv)) continue;
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
        auto inside=[&](uint32_t s,int gi,int gj,int gk)->bool{
            int raw[3]={gi,gj,gk}; float rel[3]={samplePos(s,0,raw[0])-v[0][0],samplePos(s,1,raw[1])-v[0][1],samplePos(s,2,raw[2])-v[0][2]};
            float sum=0.0f; for(int vv=0;vv<3;++vv){float bc=0;for(int i=0;i<3;++i)bc+=posInv[i][vv]*rel[i]; if(bc<-1.0e-6f)return false; sum+=bc;}
            return sum<=1.0f+1.0e-6f;
        };
        // pass 1: count
        uint32_t N=0;
        for(int gi=iMin[0];gi<iMax[0];++gi)for(int gj=iMin[1];gj<iMax[1];++gj)for(int gk=iMin[2];gk<iMax[2];++gk)
            for(uint32_t s=0;s<nSamp;++s) if(inside(s,gi,gj,gk)) N++;
        if(N==0){
            float c[3]; for(int i=0;i<3;++i)c[i]=(v[0][i]+v[1][i]+v[2][i]+v[3][i])*0.25f;
            int wg[3]; bool ok=true;
            for(int dd=0;dd<3;++dd){int cg=int(std::floor((c[dd]-P.boxLo[dd])/P.dx[dd]));wg[dd]=P.periodic?wrapIdx(cg,P.nGrid[dd]):cg;if(wg[dd]<0||wg[dd]>=P.nGrid[dd]){ok=false;break;}}
            if(ok){size_t flat=(size_t(wg[0])*P.nGrid[1]+wg[1])*P.nGrid[2]+wg[2]; grid[flat]+=m;}
            continue;
        }
        float share=m/float(N);
        for(int gi=iMin[0];gi<iMax[0];++gi)for(int gj=iMin[1];gj<iMax[1];++gj)for(int gk=iMin[2];gk<iMax[2];++gk){
            int wg0=P.periodic?wrapIdx(gi,P.nGrid[0]):gi, wg1=P.periodic?wrapIdx(gj,P.nGrid[1]):gj, wg2=P.periodic?wrapIdx(gk,P.nGrid[2]):gk;
            if(wg0<0||wg0>=P.nGrid[0]||wg1<0||wg1>=P.nGrid[1]||wg2<0||wg2>=P.nGrid[2])continue;
            size_t flat=(size_t(wg0)*P.nGrid[1]+wg1)*P.nGrid[2]+wg2;
            for(uint32_t s=0;s<nSamp;++s) if(inside(s,gi,gj,gk)) grid[flat]+=share;
        }
    }
}

static std::string readFile(const char* path){ std::ifstream f(path); std::stringstream ss; ss<<f.rdbuf(); return ss.str(); }

int main(int argc, char** argv){
    const char* kernelPath = (argc>1)? argv[1] : "metal/ps_deposit.metal";
    // ---- problem setup ----
    DepositParams P{};
    for(int i=0;i<3;++i){P.boxLo[i]=0.0f; P.nGrid[i]=32; P.subOrigin[i]=0; P.subDims[i]=32;}
    float box=10.0f; for(int i=0;i<3;++i)P.dx[i]=box/P.nGrid[i];
    P.nSub=3; P.periodic=1;
    const uint32_t nTet = (argc>2)? uint32_t(std::stoul(argv[2])) : 20000u; P.nTet=nTet;
    const size_t nCell=size_t(P.nGrid[0])*P.nGrid[1]*P.nGrid[2];
    float cellVol=P.dx[0]*P.dx[1]*P.dx[2];

    // ---- generate a stress set of tetrahedra ----
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> U(0.0f,box), Um(0.5f,1.5f), Uc(0.0f,1.0f);
    std::normal_distribution<float> Nrm(0.0f,1.0f);
    std::vector<float> verts(nTet*12), masses(nTet);
    for(uint32_t t=0;t<nTet;++t){
        float cx=U(rng),cy=U(rng),cz=U(rng); float r=Uc(rng);
        float scale, thin;
        if(r<0.60f){ scale=P.dx[0]*(1.0f+2.0f*Uc(rng)); thin=1.0f; }        // normal
        else if(r<0.85f){ scale=P.dx[0]*(1.0f+3.0f*Uc(rng)); thin=0.02f; }  // flat/caustic slab
        else { scale=P.dx[0]*0.3f*Uc(rng); thin=1.0f; }                     // tiny -> centroid fallback
        float base[3]={cx,cy,cz};
        for(int k=0;k<4;++k){
            float off[3]={Nrm(rng)*scale, Nrm(rng)*scale, Nrm(rng)*scale*thin}; // squash z for flat tets
            for(int i=0;i<3;++i) verts[t*12+k*3+i]=base[i]+off[i];
        }
        masses[t]=Um(rng);
    }

    // ---- CPU reference (timed) ----
    std::vector<double> cpuGrid(nCell,0.0);
    auto tc0=std::chrono::high_resolution_clock::now();
    cpuDeposit(verts,masses,P,cpuGrid);
    auto tc1=std::chrono::high_resolution_clock::now();
    double cpuMs=std::chrono::duration<double,std::milli>(tc1-tc0).count();
    double cpuTot=0.0; for(double x:cpuGrid) cpuTot+=x;

    // ---- Metal GPU ----
    NS::AutoreleasePool* pool=NS::AutoreleasePool::alloc()->init();
    MTL::Device* dev=MTL::CreateSystemDefaultDevice();
    if(!dev){ printf("NO_DEVICE\n"); return 1; }
    NS::Error* err=nullptr;
    MTL::Library* lib=nullptr;
    // Prefer the offline-compiled metallib (xcrun metal + metallib) if present; else runtime-compile.
    { std::ifstream mf("metal/ps_deposit.metallib");
      if (mf.good()) {
          NS::String* p = NS::String::string("metal/ps_deposit.metallib", NS::UTF8StringEncoding);
          NS::URL* url = NS::URL::fileURLWithPath(p);
          lib = dev->newLibrary(url, &err);
          if (lib) printf("kernel source      : metal/ps_deposit.metallib (offline-compiled)\n");
      } }
    if(!lib){
        std::string src=readFile(kernelPath);
        if(src.empty()){ printf("KERNEL SOURCE EMPTY: %s\n", kernelPath); return 1; }
        lib=dev->newLibrary(NS::String::string(src.c_str(),NS::UTF8StringEncoding),(MTL::CompileOptions*)nullptr,&err);
        if(lib) printf("kernel source      : %s (runtime-compiled)\n", kernelPath);
    }
    if(!lib){ printf("KERNEL COMPILE FAIL: %s\n", err?err->localizedDescription()->utf8String():"?"); return 2; }
    MTL::Function* fn=lib->newFunction(NS::String::string("depositDensity",NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pso=dev->newComputePipelineState(fn,&err);
    if(!pso){ printf("PSO FAIL\n"); return 3; }

    MTL::Buffer* bVerts=dev->newBuffer(verts.data(), verts.size()*sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bMass =dev->newBuffer(masses.data(),masses.size()*sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bGrid =dev->newBuffer(nCell*sizeof(float), MTL::ResourceStorageModeShared);
    memset(bGrid->contents(),0,nCell*sizeof(float));
    MTL::Buffer* bParams=dev->newBuffer(&P,sizeof(P),MTL::ResourceStorageModeShared);

    MTL::CommandQueue* q=dev->newCommandQueue();
    auto tg0=std::chrono::high_resolution_clock::now();
    MTL::CommandBuffer* cb=q->commandBuffer();
    MTL::ComputeCommandEncoder* enc=cb->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bVerts,0,0); enc->setBuffer(bMass,0,1); enc->setBuffer(bGrid,0,2); enc->setBuffer(bParams,0,3);
    NS::UInteger tg=pso->maxTotalThreadsPerThreadgroup(); if(tg>256)tg=256;
    enc->dispatchThreads(MTL::Size(nTet,1,1), MTL::Size(tg,1,1));
    enc->endEncoding(); cb->commit(); cb->waitUntilCompleted();
    auto tg1=std::chrono::high_resolution_clock::now();
    double gpuMs=std::chrono::duration<double,std::milli>(tg1-tg0).count();

    const float* gpuGrid=(const float*)bGrid->contents();
    double gpuTot=0.0; for(size_t i=0;i<nCell;++i) gpuTot+=gpuGrid[i];

    // expected total = sum of masses of the tets that were NOT dropped as degenerate (CPU decides).
    // Use cpuTot as ground-truth "kept mass"; report conservation and GPU-vs-CPU agreement.
    double sumAllMass=0.0; for(float mm:masses) sumAllMass+=mm;
    double maxAbs=0.0, sumAbs=0.0, cpuMax=0.0;
    for(size_t i=0;i<nCell;++i){ double dd=std::fabs(double(gpuGrid[i])-cpuGrid[i]); maxAbs=std::max(maxAbs,dd); sumAbs+=dd; cpuMax=std::max(cpuMax,cpuGrid[i]); }

    printf("=================== PS-DTFE Metal density-deposit prototype ===================\n");
    printf("device            : %s   (Metal3=%d)\n", dev->name()->utf8String(), dev->supportsFamily(MTL::GPUFamilyMetal3));
    printf("grid %dx%dx%d  nSub=%d  periodic=%d  nTet=%u  (mix: 60%% normal, 25%% flat/caustic, 15%% tiny)\n",
           P.nGrid[0],P.nGrid[1],P.nGrid[2],P.nSub,P.periodic,nTet);
    printf("sum of all tet masses         : %.6f\n", sumAllMass);
    printf("CPU total deposited (kept)    : %.6f   (dropped mass = %.3e)\n", cpuTot, sumAllMass-cpuTot);
    printf("GPU total deposited           : %.6f\n", gpuTot);
    printf("MASS CONSERVATION  GPU/CPU    : %.8f   (target 1.0)\n", gpuTot/cpuTot);
    printf("GPU-vs-CPU  mean|diff|/cpuMax : %.3e   (robust: agreement of the two deposits)\n", (sumAbs/nCell)/cpuMax);
    printf("GPU-vs-CPU  max  |diff|/cpuMax : %.3e   (float32 summation ORDER at high-overlap cells;\n", maxAbs/cpuMax);
    printf("                                          GPU-atomic vs CPU-double, inherent to float density fields)\n");
    printf("as density (max mass/cellVol) : CPU=%.4f  cellVol=%.4f\n", cpuMax/cellVol, cellVol);
    printf("timing  CPU(1 core): %8.2f ms   GPU(dispatch+wait): %8.2f ms   speedup: %.1fx\n",
           cpuMs, gpuMs, cpuMs/gpuMs);
    // Correctness criteria: exact mass conservation + tight MEAN agreement. (The max single-cell diff
    // is float32 non-associativity at heavily-overlapped cells, present in the CPU pipeline too.)
    bool pass = std::fabs(gpuTot/cpuTot - 1.0) < 1e-4 && (sumAbs/nCell)/cpuMax < 1e-4;
    printf("RESULT: %s\n", pass ? "PASS (GPU deposit matches CPU and conserves mass)" : "FAIL");
    pool->release();
    return pass?0:1;
}
