# CosmoDTFE side of the comparison. One process per estimator family, so each peak RSS is its own.
#   julia -t 10 --project=<env> run_cosmo.jl <dtfe|ps|tune> <snap.h5> <outroot> <npix> <z0> <grid> <depth> <pad> [lo hi periodic nbar]
# lo/hi: the region (slice and grid span [lo, hi] on every axis; default the box); periodic 0 = the
# plain (non-periodic) estimators; nbar = the mean number density for rho/rho_bar (default N/L^3).
# Writes <outroot>.<field>.bin (float32 / int32, C order: slice [ix][iy], grid [ix][iy][iz]) and one
# JSON line per stage to <outroot>.timing.jsonl.
using CosmoDTFE, HDF5, Printf

function loadSnap(path)
    h5open(path, "r") do f
        hdr = attributes(f["Header"])
        L = read(hdr["BoxSize"])
        x = read(f["PartType1/Coordinates"])                # 3 x N (column-major)
        q = read(f["PartType1/InitialCoordinates"])
        v = read(f["PartType1/Velocities"])
        toP(m) = [Point3(Float64(m[1, i]), Float64(m[2, i]), Float64(m[3, i])) for i in axes(m, 2)]
        return Float64(L), toP(x), toP(q), toP(v)
    end
end

const LOG = Ref{IOStream}()
function stage(name, t; extra...)
    s = "{\"stage\":\"$name\",\"seconds\":$(round(t, digits=3)),\"maxrss_gb\":$(round(Sys.maxrss() / 2^30, digits=3))"
    for (k, v) in extra
        s *= ",\"$k\":" * (v isa AbstractString ? "\"$v\"" : string(v))
    end
    println(LOG[], s * "}")
    flush(LOG[])
    @printf("  %-14s %8.2f s   peak RSS %6.2f GB\n", name, t, Sys.maxrss() / 2^30)
end

writeC(path, a::AbstractArray{T,2}) where T = write(path, permutedims(a, (2, 1)))        # [ix][iy]
writeC(path, a::AbstractArray{T,3}) where T = write(path, permutedims(a, (3, 2, 1)))     # [ix][iy][iz]

# a threaded map over a list of points (the library threads its own grid calls the same way)
function tmap(f, pts, T)
    out = Vector{T}(undef, length(pts))
    Threads.@threads for i in eachindex(pts)
        out[i] = f(pts[i])
    end
    out
end

function main(mode, path, outroot, npix, z0, G, depth, pad; lo=nothing, hi=nothing, periodic=true, nbar=0.0)
    L, x, q, v = loadSnap(path)
    N = length(x)
    box = Point3(L, L, L)
    rhobar = nbar > 0 ? nbar : N / L^3                   # unit weights: DTFE density in number/volume
    a, b = something(lo, 0.0), something(hi, L)
    sx = range(a + (b - a) / npix / 2, b - (b - a) / npix / 2, length=npix)
    gx = range(a + (b - a) / max(G, 1) / 2, b - (b - a) / max(G, 1) / 2, length=max(G, 1))
    if mode == "dtfe"
        t = @elapsed est = periodic ? PeriodicEstimator(DensityEstimator, x; boxSize=box, padding=pad, depth=depth) :
                                      DensityEstimator(x; depth=depth)
        stage("build", t; N=N, depth=depth, pad=pad)
        t = @elapsed s = est((sx, sx, [z0]))
        stage("slice", t; points=npix^2)
        writeC(outroot * ".rho_slice.bin", Float32.(s[:, :, 1] ./ rhobar))
        if G > 0
            t = @elapsed g = est((gx, gx, gx))
            stage("grid", t; points=G^3)
            writeC(outroot * ".rho_grid.bin", Float32.(g ./ rhobar))
        end
    elseif mode == "ps"
        t = @elapsed est = periodic ? PeriodicPhaseSpaceEstimator(q, x, v; boxSize=box, padding=pad, depth=depth) :
                                      PhaseSpaceEstimator(q, x, v; depth=depth)
        stage("build", t; N=N, depth=depth, pad=pad)
        spts = [Point3(a, b, z0) for a in sx for b in sx]          # ix major, iy fastest
        t = @elapsed begin
            vs = tmap(p -> est(p), spts, Point3)                       # sum over streams (the library's output)
            ns = tmap(p -> streamNumber(est, p), spts, Int)
        end
        stage("slice", t; points=npix^2)
        write(outroot * ".streams_slice.bin", Int32.(ns))
        write(outroot * ".vsum_slice.bin", Float32.(reduce(hcat, vs)))      # 3 x n, point-major
        if G > 0
            gpts = [Point3(a, b, c) for a in gx for b in gx for c in gx]
            t = @elapsed begin
                vg = tmap(p -> est(p), gpts, Point3)
                ng = tmap(p -> streamNumber(est, p), gpts, Int)
            end
            stage("grid", t; points=G^3)
            write(outroot * ".streams_grid.bin", Int32.(ng))
            write(outroot * ".vsum_grid.bin", Float32.(reduce(hcat, vg)))
        end
    elseif mode == "tune"
        # BVH depth scan for the PS estimator on one tessellation (the constructor's own pieces):
        # time a 128^2 probe of the slice per depth; the public constructor is used for the real run
        cs, cx, cv = CosmoDTFE.periodicPhaseSpaceCopies(q, x, v, Point3(0.0, 0.0, 0.0), box, pad)
        t = @elapsed tr = tessellate(cs)
        tets = tr[2]
        stage("tessellate", t; N=length(cs))
        tri = CosmoDTFE.Triangulation3D{Point3}(cx, cv)
        probe = [Point3(a, b, z0) for a in range(0.5, L - 0.5, length=128) for b in range(0.5, L - 0.5, length=128)]
        for d in (9, 16, 18, 20, 21, 22, 23, 24)
            tb = @elapsed bvh = BoundingVolumeHierarchy(cx, tets, d)
            e = CosmoDTFE.PhaseSpaceEstimator{Point3}(bvh, tri, tets)
            tmap(p -> streamNumber(e, p), probe[1:64], Int)                   # warm
            tp = @elapsed tmap(p -> streamNumber(e, p), probe, Int)
            stage("depth$d", tp; bvh_seconds=round(tb, digits=2), probe_points=length(probe))
            bvh = e = nothing
            GC.gc()
        end
    end
end

mode, path, outroot = ARGS[1], ARGS[2], ARGS[3]
npix, z0, G, depth, pad = parse(Int, ARGS[4]), parse(Float64, ARGS[5]), parse(Int, ARGS[6]), parse(Int, ARGS[7]), parse(Float64, ARGS[8])
kw = length(ARGS) >= 12 ? (lo=parse(Float64, ARGS[9]), hi=parse(Float64, ARGS[10]), periodic=ARGS[11] == "1", nbar=parse(Float64, ARGS[12])) : NamedTuple()
LOG[] = open(outroot * ".timing.jsonl", "a")
# JIT warm-up on a tiny snapshot, so compile time is not billed to the benchmark
warm = joinpath(dirname(path), "warm.h5")
if isfile(warm) && mode != "tune"
    LOG0 = LOG[]; LOG[] = open(tempname(), "w")
    redirect_stdout(devnull) do
        main(mode, warm, tempname(), 8, 55.0, 4, 6, pad; kw...)      # the same code path as the real run
    end
    close(LOG[]); LOG[] = LOG0
end
println("CosmoDTFE $mode  $(basename(path))  threads=$(Threads.nthreads())")
tw = @elapsed main(mode, path, outroot, npix, z0, G, depth, pad; kw...)
stage("total", tw)
close(LOG[])
