/* Replaced global operator new/delete routing >= threshold allocations to mmap'ed scratch
 * files. See scratch_alloc.h for the design contract. LINKED INTO THE BINARIES ONLY -- libDTFE
 * links scratch_alloc_stub.cc instead (a library must not hijack its host's allocator).
 *
 * Re-entrancy note: everything here must be allocation-free. The registry is a fixed
 * malloc-free slot table (a std::unordered_map would call operator new under our own mutex ->
 * deadlock), filenames are built with snprintf into stack buffers, and no std::string is
 * constructed anywhere on the allocation path.
 */

#include "scratch_alloc.h"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

namespace
{
    std::atomic<bool> g_enabled{false};
    std::size_t       g_threshold = ~std::size_t(0);
    char              g_dir[1024] = {0};
    std::atomic<long> g_liveBytes{0};
    std::atomic<long> g_totalAllocs{0};

    // Registry of live scratch mappings. Full-grid vectors number in the tens, so a fixed
    // table with linear scan is plenty -- and it never allocates. If it ever fills, the
    // allocation falls back to the heap (correct, just not disk-backed).
    //
    // EVERYTHING here must be trivially destructible: operator delete keeps being called
    // during exit-time static destruction of OTHER translation units, whose order relative
    // to this one is unspecified. A std::mutex has a destructor -- locking it after that ran
    // is UB (measured: SIGSEGV at exit teardown) -- so the lock is a pthread mutex with a
    // static initializer, which is plain data and never dies.
    //
    // The delete path must NOT take the mutex for ordinary heap pointers: every delete in the
    // process runs through here, and a global lock serialized all concurrent partitions -- the
    // PS vertex-density pass (millions of small vector frees) went from 10 s to 191 s per
    // partition at --max-concurrent 5. So each slot's pointer is an atomic, published with
    // release ordering BEFORE the mapping is handed out, and freeImpl first rules a pointer out
    // lock-free: outside [g_lo, g_hi) (the span of every mapping ever made) it cannot be ours,
    // and otherwise a scan of the slots ever used finds it or not. The mutex is taken only to
    // create or unmap a mapping -- tens of times per run. A pointer being deleted was returned
    // by operator new earlier (happens-before), so the scan always sees its own slot.
    struct Slot { std::atomic<void*> p; std::size_t sz; };   // std::atomic<void*>: trivially destructible
    constexpr int kSlots = 512;
    Slot            g_slots[kSlots] = {};
    std::atomic<int>            g_slotHigh{0};                      // one past the highest slot ever used
    std::atomic<std::uintptr_t> g_lo{~std::uintptr_t(0)}, g_hi{0};  // address span of all mappings made
    pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;
    std::atomic<std::uint64_t> g_counter{0};
    std::atomic<bool> g_warnedFallback{false};

    struct MutexGuard
    {
        explicit MutexGuard(pthread_mutex_t &m) : m_(m) { pthread_mutex_lock(&m_); }
        ~MutexGuard() { pthread_mutex_unlock(&m_); }
        pthread_mutex_t &m_;
    };

    inline std::size_t roundToPage(std::size_t sz)
    {
        std::size_t const page = std::size_t( sysconf(_SC_PAGESIZE) );
        return (sz + page - 1) / page * page;
    }

    void *scratchTryMap(std::size_t sz)
    {
        if ( not g_enabled.load(std::memory_order_acquire) or sz < g_threshold )
            return nullptr;

        std::size_t const mapSz = roundToPage(sz);

        // refuse to start a mapping the volume cannot hold (2 GB safety margin): a full disk
        // later means SIGBUS on a dirty-page writeback, which is far worse than a heap fallback
        struct statvfs vfs;
        if ( statvfs(g_dir, &vfs) == 0 )
        {
            double const freeBytes = double(vfs.f_bavail) * double(vfs.f_frsize);
            if ( freeBytes < double(mapSz) + 2.e9 )
            {
                if ( not g_warnedFallback.exchange(true) )
                    std::fprintf( stderr, "SCRATCH: only %.1f GB free in '%s' but %.1f GB requested; "
                                          "falling back to heap for this and any further oversized "
                                          "allocation that does not fit (expect swapping).\n",
                                  freeBytes/1.e9, g_dir, mapSz/1.e9 );
                return nullptr;
            }
        }

        char path[1200];
        std::snprintf( path, sizeof(path), "%s/dtfe_scratch_%ld_%llu.mem",
                       g_dir, long(getpid()),
                       (unsigned long long) g_counter.fetch_add(1) );
        int const fd = ::open( path, O_CREAT | O_EXCL | O_RDWR, 0600 );
        if ( fd < 0 )
            return nullptr;
        if ( ::ftruncate( fd, off_t(mapSz) ) != 0 )
        {
            ::close(fd); ::unlink(path);
            return nullptr;
        }
        void *p = ::mmap( nullptr, mapSz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0 );
        // the name is dropped IMMEDIATELY: the mapping keeps the storage alive, and the kernel
        // reclaims it when the process exits -- normally or by crash. No stale files possible.
        ::unlink(path);
        ::close(fd);
        if ( p == MAP_FAILED )
            return nullptr;

        {
            MutexGuard lock(g_mtx);
            for (int i = 0; i < kSlots; ++i)
                if ( g_slots[i].p.load(std::memory_order_relaxed) == nullptr )
                {
                    g_slots[i].sz = mapSz;
                    std::uintptr_t const a = reinterpret_cast<std::uintptr_t>(p);
                    if ( a < g_lo.load(std::memory_order_relaxed) )
                        g_lo.store(a, std::memory_order_release);
                    if ( a + mapSz > g_hi.load(std::memory_order_relaxed) )
                        g_hi.store(a + mapSz, std::memory_order_release);
                    if ( i + 1 > g_slotHigh.load(std::memory_order_relaxed) )
                        g_slotHigh.store(i + 1, std::memory_order_release);
                    g_slots[i].p.store(p, std::memory_order_release);   // publish last
                    g_liveBytes  += long(mapSz);
                    g_totalAllocs += 1;
                    return p;
                }
        }
        ::munmap(p, mapSz);   // table full: extremely unlikely, but stay correct
        return nullptr;
    }

    // true (+ unmap) when p was one of ours; false = ordinary heap pointer
    bool scratchRelease(void *p)
    {
        if ( g_totalAllocs.load(std::memory_order_relaxed) == 0 )
            return false;     // nothing ever mapped
        // lock-free: outside every mapping ever made, or in no used slot => an ordinary heap pointer
        std::uintptr_t const a = reinterpret_cast<std::uintptr_t>(p);
        if ( a < g_lo.load(std::memory_order_acquire) or a >= g_hi.load(std::memory_order_acquire) )
            return false;
        int const high = g_slotHigh.load(std::memory_order_acquire);
        bool ours = false;
        for (int i = 0; i < high and not ours; ++i)
            ours = ( g_slots[i].p.load(std::memory_order_acquire) == p );
        if ( not ours )
            return false;
        MutexGuard lock(g_mtx);
        for (int i = 0; i < kSlots; ++i)
            if ( g_slots[i].p.load(std::memory_order_relaxed) == p )
            {
                ::munmap( p, g_slots[i].sz );
                g_liveBytes -= long(g_slots[i].sz);
                g_slots[i].p.store(nullptr, std::memory_order_release);
                g_slots[i].sz = 0;
                return true;
            }
        return false;
    }

    inline void *allocImpl(std::size_t sz)
    {
        if ( void *p = scratchTryMap(sz) )
            return p;
        return std::malloc( sz ? sz : 1 );
    }

    inline void *allocAlignedImpl(std::size_t sz, std::size_t al)
    {
        // mmap memory is page-aligned, satisfying any alignment up to the page size
        if ( al <= std::size_t(sysconf(_SC_PAGESIZE)) )
            if ( void *p = scratchTryMap(sz) )
                return p;
        void *p = nullptr;
        if ( posix_memalign( &p, al < sizeof(void*) ? sizeof(void*) : al, sz ? sz : 1 ) != 0 )
            return nullptr;
        return p;
    }

    inline void freeImpl(void *p)
    {
        if ( p == nullptr ) return;
        if ( scratchRelease(p) ) return;
        std::free(p);   // covers malloc AND posix_memalign pointers
    }
}

namespace ScratchAlloc
{
    bool scratchArm(char const *dir, std::size_t thresholdBytes)
    {
        std::snprintf( g_dir, sizeof(g_dir), "%s", dir );
        g_threshold = thresholdBytes;
        g_enabled.store( true, std::memory_order_release );
        return true;
    }
    long scratchLiveBytes()   { return g_liveBytes.load(); }
    long scratchTotalAllocs() { return g_totalAllocs.load(); }
}

// ------------------------- replaced global allocation functions -------------------------
// The full replaceable set: plain, nothrow, array, sized-delete and aligned variants. Every
// delete funnels through freeImpl's registry check, so any new/delete pairing is safe.

void *operator new(std::size_t sz)
{
    if ( void *p = allocImpl(sz) ) return p;
    throw std::bad_alloc();
}
void *operator new[](std::size_t sz)
{
    if ( void *p = allocImpl(sz) ) return p;
    throw std::bad_alloc();
}
void *operator new(std::size_t sz, std::nothrow_t const &) noexcept  { return allocImpl(sz); }
void *operator new[](std::size_t sz, std::nothrow_t const &) noexcept{ return allocImpl(sz); }

void *operator new(std::size_t sz, std::align_val_t al)
{
    if ( void *p = allocAlignedImpl(sz, std::size_t(al)) ) return p;
    throw std::bad_alloc();
}
void *operator new[](std::size_t sz, std::align_val_t al)
{
    if ( void *p = allocAlignedImpl(sz, std::size_t(al)) ) return p;
    throw std::bad_alloc();
}
void *operator new(std::size_t sz, std::align_val_t al, std::nothrow_t const &) noexcept
{ return allocAlignedImpl(sz, std::size_t(al)); }
void *operator new[](std::size_t sz, std::align_val_t al, std::nothrow_t const &) noexcept
{ return allocAlignedImpl(sz, std::size_t(al)); }

void operator delete(void *p) noexcept                                   { freeImpl(p); }
void operator delete[](void *p) noexcept                                 { freeImpl(p); }
void operator delete(void *p, std::size_t) noexcept                      { freeImpl(p); }
void operator delete[](void *p, std::size_t) noexcept                    { freeImpl(p); }
void operator delete(void *p, std::nothrow_t const &) noexcept           { freeImpl(p); }
void operator delete[](void *p, std::nothrow_t const &) noexcept         { freeImpl(p); }
void operator delete(void *p, std::align_val_t) noexcept                 { freeImpl(p); }
void operator delete[](void *p, std::align_val_t) noexcept               { freeImpl(p); }
void operator delete(void *p, std::size_t, std::align_val_t) noexcept    { freeImpl(p); }
void operator delete[](void *p, std::size_t, std::align_val_t) noexcept  { freeImpl(p); }
