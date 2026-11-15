// include/basalt/core/waitpkg.hpp - waits that leave the core in a light sleep (Intel WAITPKG: UMONITOR, UMWAIT, TPAUSE).
//
// A thread that waits a few hundred microseconds for another one cannot block in the OS (the wake-up costs more than
// the wait), so it spins; PAUSE keeps its core at full clock and voltage meanwhile.  Where the processor has WAITPKG
// the wait can sit in C0.2 (C0.1 on request) instead: the core stops executing until a monitored cache line is
// written or a TSC deadline passes, and runs again about a microsecond later.  On a machine whose processor and GPU
// share a cooler or a power budget, the power a waiting core does not draw is the GPU's.
//
// The OS bounds one wait (Linux: /sys/devices/system/cpu/umwait_control), so every wait here is a slice the caller
// repeats; a wait is never the only thing between a thread and its work.
#pragma once

#include <atomic>
#include <cstdint>

#include "basalt/platform/compiler.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#define BASALT_HAVE_WAITPKG 1
#if defined(__GNUC__) || defined(__clang__)
#include <x86intrin.h>
#define BASALT_TARGET_WAITPKG __attribute__((target("waitpkg")))
#else
#define BASALT_TARGET_WAITPKG
#endif
#else
#define BASALT_HAVE_WAITPKG 0
#endif

namespace basalt::core {

/// Whether the processor runs UMONITOR, UMWAIT and TPAUSE (CPUID.(EAX=7,ECX=0):ECX[5]).
inline bool waitpkg_supported() {
#if BASALT_HAVE_WAITPKG
    unsigned r[4] = {0, 0, 0, 0};
    basalt::platform::cpuid(0, 0, r);
    if (r[0] < 7) return false;
    basalt::platform::cpuid(7, 0, r);
    return ((r[2] >> 5) & 1u) != 0;
#else
    return false;
#endif
}

#if BASALT_HAVE_WAITPKG
/// One slice of a wait for `word` to change from `seen`: at most `tsc_span` TSC ticks in C0.2 (C0.1 with `c01`), ended
/// early by a write to the word's cache line.  The word is read again after the monitor is armed, so a write between
/// the caller's check and the wait is not slept through.  Only where waitpkg_supported().
BASALT_TARGET_WAITPKG inline void umwait_while(const std::atomic<uint32_t>& word, uint32_t seen, bool c01,
                                               unsigned long long tsc_span) {
    _umonitor((void*) &word);
    if (word.load(std::memory_order_acquire) != seen) return;
    (void) _umwait(c01 ? 1u : 0u, __rdtsc() + tsc_span);
}

/// A timed pause of at most `tsc_span` TSC ticks in C0.2 (C0.1 with `c01`), for a wait on memory another device
/// writes, which no monitor sees.  Only where waitpkg_supported().
BASALT_TARGET_WAITPKG inline void tpause_for(unsigned long long tsc_span, bool c01 = false) {
    (void) _tpause(c01 ? 1u : 0u, __rdtsc() + tsc_span);
}
#endif

}  // namespace basalt::core
