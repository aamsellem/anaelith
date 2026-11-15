# anaelith

A fork of [Basalt](https://github.com/jesdga95/basalt) (MIT) worked on an Olares One: an RTX 5090 Laptop GPU with
24 GB, a Core Ultra 9 275HX (8 performance cores, 16 efficient cores, no SMT) and 96 GB of DDR5 in a mini PC. The
changes are for machines of that kind: one 24 GB card, a hybrid Intel processor, and a processor and a GPU that share
a cooler and a power budget.

Branch `one` starts from upstream 0.4.0. Upstream's README, documentation and licence apply unchanged; this page only
lists what differs. Binaries keep the name `basalt`.

## What differs from upstream

### Hybrid Intel processors are recognized on Linux

Upstream reads a core's class from `/sys/devices/system/cpu/cpuN/cpu_capacity`. x86 kernels do not publish that file
(it comes from the arch_topology driver, which x86 does not build), so a hybrid Intel processor counted as one class
and the pool started with every core but one: 23 workers on a 275HX. The fork falls back to the CPU lists of the
kernel's two hybrid PMUs, `/sys/devices/cpu_core/cpus` and `/sys/devices/cpu_atom/cpus` (Linux 5.13 and later), and
the existing rule then gives the performance cores minus one plus half the efficient cores: 15 here.

Measured on this machine with upstream 0.3.3 (UD-IQ3_XXS, my own prompts, 800 tokens each): 23 workers 149.3 tok/s
greedy and 136.0 sampled, 15 workers 164.3 and 167.8. `tests/platform/cpu_test.cpp` checks the topology against the
kernel's lists where they exist.

### The CPU tier's warm-up runs on packs whose layers differ in format

Between two doorbells the idle pool scores the next layer's router and reads the likeliest CPU-held expert into its
caches. Upstream turns this off unless every layer's blobs have the same size, which leaves out every Unsloth Dynamic
pack (UD-IQ3_XXS mixes IQ2_S, IQ3_XXS and IQ4_XS gate and up rows with IQ4_NL and Q2_0 down rows). The fork keeps a
size and a down-row offset per layer.

`--exp warm_n=N` sets how many predicted experts are read, likeliest first (default 1, at most 4, 0 turns the warm-up
off), and `--decode-timing` prints how the predictions turned out.

Measured (UD-IQ3_XXS, `basalt-bench`, six runs per setting): the prediction is right for 43 to 51% of its picks on
`conv`, 32 to 40% on `prose` and 15% on `struct`. The CPU's time per expert falls 2% with one expert read and 5% with
three on `conv`. End to end that is +2% on `conv` at best and nothing on the other fixtures, inside the run-to-run
noise. It repairs a feature that was off by accident; it is not a speed-up on this machine.

### Waiting threads can sleep lightly (`--exp umwait`, `--exp host_tpause`)

The pool's workers wait for the GPU about two thirds of the time during decode, pausing at full clock. With
`--exp umwait=2` a parked worker arms UMONITOR on the word the host writes to publish a batch and waits in C0.2 with
UMWAIT (`umwait=1`: C0.1). With `--exp host_tpause=1` the host thread pauses in C0.2 for about a microsecond between
two polls of the GPU's ring (TPAUSE). Both need WAITPKG and do nothing without it. Off by default.

Measured (order A B C C B A, three repetitions of the three fixtures per arm, package power from RAPL and GPU power
from NVML while the GPU is busy):

| | CPU package | GPU | SM clock | struct tok/s |
|---|---:|---:|---:|---:|
| pause (upstream) | 61.6 W | 137.3 W | 1930 MHz | 285.5 |
| workers in C0.2 | 52.3 W | 142.4 W | 1981 MHz | 289.0 |
| workers in C0.2, host TPAUSE | 50.6 W | 142.8 W | 1988 MHz | 289.0 |

The processor draws 11 W less and the GPU takes about half of it back at the same temperature. `struct`, the only
fixture whose text is the same on every run, gains nothing on a cold first repetition and 2 to 2.7% on the next two.
The CPU's time per expert does not get worse.

## How things were measured

`bin/basalt-bench --model PACK -t conv,struct,prose -r N -o json --logs DIR -- FLAGS --decode-timing`, always with
`--resident-budget-gib 48 --ram-reserve-mib 12288 --pool-workers 15`. This GPU throttles at 85 C and loses 15% of its
clock between a cold and a hot repetition, so arms alternate in a balanced order, each starts once the GPU is back at
54 C or after 150 s, and a setting is judged on both of its arms. The same binary still varies by a few percent on the
sampled fixtures.

## Tried and not kept

- Exact speculative sampling for Flash-Next's MTP drafter. `--exp spec_log=1` gives what is needed to estimate it:
  today 78.1% of drafts are kept on the sampled fixtures, and drawing drafts from the head's top four with the
  min(1, p/q) rule would keep 78.9 to 79.5%. The head is overconfident; when it is wrong it gives the right token
  almost no mass.
- Fewer pool workers: 5, 7 and 11 lose to 15 end to end, although the GPU clocks 5 to 9% higher with 5.
