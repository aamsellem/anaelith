# anaelith

anaelith is a fork of [Basalt](https://github.com/jesdga95/basalt), a local inference engine for mixture-of-experts
models on NVIDIA Blackwell GPUs. The fork is worked on and measured on one machine, an Olares One: a mini PC with an
RTX 5090 Laptop GPU (24 GB), a Core Ultra 9 275HX (8 performance cores and 16 efficient cores, no SMT) and 96 GB of
DDR5. Its changes are for machines of that kind: one 24 GB card, a hybrid Intel processor, and a processor and a GPU
that share a cooler and a power budget.

Everything else is Basalt as upstream wrote it: the models, the `.basalt` packs, the server with its OpenAI and
Anthropic APIs, the flags, and the documentation under [docs](docs). The binaries keep the name `basalt`. The fork
publishes no release archives and no packs; upstream's are at
[its releases](https://github.com/jesdga95/basalt/releases) and on
[Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt).

## Branches

- `one`: the fork. It starts from upstream 0.4.0.
- `main`: upstream's `main` as it was when the fork was made, unchanged.
- `tier-across-sizes`: a measured change that did not win and is not in `one` (see [below](#tried-and-not-kept)).

## What differs from upstream

All figures on this page were measured on the One with Qwen3.8-Flash-Next, pack `Qwen3.8-FN-UD-IQ3_XXS`, the GPU's
power limit at 175 W. During decode the GPU sits at its thermal limit (86 C, fans at their maximum) and draws 130 to
150 W.

### Hybrid Intel processors are recognized on Linux

Upstream reads a core's class from `/sys/devices/system/cpu/cpuN/cpu_capacity`. x86 kernels do not publish that file
(it comes from the arch_topology driver, which x86 does not build), so a hybrid Intel processor counted as one class
and the pool started with every core but one: 23 workers on a 275HX. The fork falls back to the CPU lists of the
kernel's two hybrid PMUs, `/sys/devices/cpu_core/cpus` and `/sys/devices/cpu_atom/cpus` (Linux 5.13 and later), and
the documented rule then gives the performance cores minus one plus half the efficient cores: 15 here, with no
`--pool-workers`.

Measured with upstream 0.3.3 (800 tokens a prompt): 23 workers 149.3 tok/s greedy and 136.0 sampled, 15 workers
164.3 and 167.8. `tests/platform/cpu_test.cpp` checks the topology against the kernel's lists where they exist.

### A tool call that quotes tool-call markup is read as it is written

A model that edits a chat template or a parser writes a line of tool-call markup as an argument, for instance
`<parameter=limit>25</parameter>` as the new text of an edit. It then often takes the quoted closing tag for its
value's own and writes the next parameter or the function's end straight after it. Upstream's parser ended the value
before that tag (`<parameter=limit>25`), and its grammar refused the function's end at its last character, where the
model could only add a stray one (`</function≥`).

- The parser (`serve/frontend.py`, `param_close`) follows the parameters a value quotes. The tag that closes a
  quoted one belongs to the value; where the call's structure follows it directly and the tag does not start a line,
  the value ends with it. After a quoted parameter, a parameter the function does not declare or already has, and a
  function end that the call's end does not follow, stay in the value. Nothing is added to the model's text or
  changed in it: the rule only decides which closing tag ends the value.
- The grammar (`src/program/grammar/schema.cpp`) lets a text value hold its closing tag and close at the tag with or
  without the line break before it. It follows the quoted parameters and lets the text go on after their tag,
  except with a function end that the call's end or a new call follows.

Measured with 40 two-turn sequences that ask for an edit of a template line and a test run, checked on whether the
edit gives the right file and both calls are made:

| | sequences right, of 40 |
|---|---:|
| upstream parser, no grammar | 15 |
| fork's parser, no grammar | 27 |
| fork's parser and grammar | 30 |

What is left is the model's own: it makes one call of the two (9 to 12 sequences), or writes the tags as HTML
entities. The grammar's last rule, the new call after a function end, was added after these runs. Tests:
`serve/tests/test_quoted_parameter_values.py`, `tests/program/grammar_schema_test.cpp`.

### The CPU tier's warm-up runs on packs whose layers differ in format

Between two doorbells the idle pool scores the next layer's router and reads the likeliest CPU-held expert into its
caches. Upstream turns this off unless every layer's blobs have the same size, which leaves out every Unsloth Dynamic
pack (UD-IQ3_XXS mixes IQ2_S, IQ3_XXS and IQ4_XS gate and up rows with IQ4_NL and Q2_0 down rows). The fork keeps a
size and a down-row offset per layer. `--exp warm_n=N` sets how many predicted experts are read, likeliest first
(default 1, at most 4, 0 turns the warm-up off), and `--decode-timing` prints how the predictions turned out.

Measured: the prediction is right for 43 to 51% of its picks on `conv`, 32 to 40% on `prose` and 15% on `struct`.
End to end that is +2% on `conv` at best, inside the run-to-run noise. It repairs a feature that was off by accident;
it is not a speed-up on this machine.

### Waiting threads can sleep lightly (`--exp umwait`, `--exp host_tpause`)

The pool's workers wait for the GPU about two thirds of the time during decode, pausing at full clock. With
`--exp umwait=2` a parked worker arms UMONITOR on the word the host writes to publish a batch and waits in C0.2 with
UMWAIT (`umwait=1`: C0.1). With `--exp host_tpause=1` the host thread pauses in C0.2 for about a microsecond between
two polls of the GPU's ring (TPAUSE). Both need WAITPKG and do nothing without it. Off by default.

| | CPU package | GPU | SM clock | struct tok/s |
|---|---:|---:|---:|---:|
| pause (upstream) | 61.6 W | 137.3 W | 1930 MHz | 285.5 |
| workers in C0.2 | 52.3 W | 142.4 W | 1981 MHz | 289.0 |
| workers in C0.2, host TPAUSE | 50.6 W | 142.8 W | 1988 MHz | 289.0 |

The processor draws 11 W less and the GPU takes about half of it back at the same temperature. `struct` gains nothing
on a cold first repetition and 2 to 2.7% on the next two.

## Build and run

The requirements and the build are upstream's ([building](docs/building.md), [platforms](docs/platforms.md)): Linux
x86-64 with AVX2 or Windows x64, a Blackwell GPU with CUDA 13, CMake 3.24+, GCC 13+, Python 3.10+.

```sh
git clone -b one https://github.com/aamsellem/anaelith && cd anaelith
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j                  # bin/basalt, bin/basalt-bench, bin/basalt-probe, bin/basalt-vision
                                        # one model only: -DBASALT_MODELS=qwen4exp, deepseek41 or qwen35
pip install -r requirements.txt
hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-UD-IQ3_XXS.basalt --local-dir models
```

On a 24 GB card with 96 GB of RAM:

```sh
bin/basalt serve models/Qwen3.8-FN-UD-IQ3_XXS.basalt --port 8080 --max-context 65536 --kv int8 \
    --resident-budget-gib 48 --ram-reserve-mib 12288
```

The server listens on 127.0.0.1. Every setting: [configuration](docs/configuration/index.md). The models and their
guides: [models](docs/models/index.md).

## How things were measured

`bin/basalt-bench --model PACK -t conv,struct,prose -r N -o json --logs DIR -- FLAGS --decode-timing`, with
`--resident-budget-gib 48 --ram-reserve-mib 12288 --pool-workers 15`. `conv` and `prose` are sampled and differ from
run to run. `struct` is greedy and gives the same 7,230 tokens every time, so it is the fixture a small difference
can be read on.

The GPU loses 15% of its clock between a cold and a hot repetition. Two settings are therefore compared in one
session in a balanced order (A B B A), each arm starts once the GPU is back near 55 C or after a fixed wait, and a
setting is judged on both of its arms. The processor's package power comes from RAPL and the GPU's power and clock
from NVML, read every second while the GPU is busy.

Upstream 0.4.0 on this machine, two sessions:

| | tok/s |
|---|---:|
| `conv` | 178.4, 179.3 |
| `struct` | 301.4, 301.6 |
| `prose` | 151.6, 153.6 |
| prompt read, 16K tokens | 2,780, 2,743 |
| prompt read, 32K tokens | 2,774, 2,748 |

## What sets the speed on this machine

- **The window.** A verify window of T token rows takes about 9.3 + 1.9 T ms: 16 ms at T 3.6 on `conv`, 24 ms at
  T 7.7 on `struct`, for 2.8 and 7.4 tokens. The first term is the pass over the dense weights, shared by the rows;
  each row then reads its own experts.
- **The GPU's clock.** At its thermal limit the GPU runs its SM clock between 1.7 and 2.6 GHz, 2.0 GHz on average,
  of 3.09. Speed follows the clock, which is why a watt less in the processor shows. Heat builds up: three
  repetitions of `struct` back to back give 306, 292 and 270 tok/s.
- **The experts on the GPU.** The stage profile (`--profile`) puts the GPU's own experts at 5.5 of 16.8 ms a window
  on `conv` and 12.7 of 27.1 ms on `struct`, the largest stage in both. At about 5 us for an expert of 1.9 MiB that
  is 40% of the card's memory bandwidth: these kernels are bound by decoding the quantized weights, at the clock the
  heat allows.
- **Not the CPU tier.** Three changes cut the CPU's expert work and left the speed where it was: the tier ranked
  across layers (18% fewer experts on the CPU in `conv`), every expert of layer 0 resident (that layer's wait on the
  CPU gone), and huge pages under the experts in RAM. Fewer workers do cost speed, so the tier is sized about right.
- **The drafts.** On sampled text 78% of the MTP's drafts are kept. Drawing the drafts and keeping them with the
  min(1, p/q) rule would keep 79 to 79.5%: the head is confident where the model is not. Upstream's
  `--relaxed-drafting`, which changes what is sampled, keeps more: `conv` 167.0 to 173.2 tok/s and `prose` 158.2 to
  167.0 at 0.5, 183.7 and 167.5 at 0.3 (six runs each).

## Tried and not kept

- **The VRAM tier ranked across layers** on a pack whose layers differ in blob size (upstream does it where they do
  not). Branch `tier-across-sizes`, `--exp tier_across=1`. Experts on the CPU per layer and window: `conv` 3.30 to
  2.70, `prose` 2.05 to 1.65, `struct` 4.64 to 4.21; package power 52.4 to 50.2 W; speed unchanged (`conv` 168.0 to
  167.4 tok/s, `struct` 297.2 to 298.3).
- **Every expert of layer 0 resident.** In `conv` and `prose` the first three layers route the most varied experts
  and the device waits longest for the CPU there (0.43 ms a window for layer 0). With layer 0 whole in VRAM that
  wait is zero and the speed the same: 166.2 to 166.7 tok/s on `conv`, 158.1 to 158.4 on `prose`, six runs each.
- **Huge pages under the experts in RAM.** The arena is aligned for them but not requested, so with transparent
  huge pages in `madvise` mode it sits on 4 KiB pages. Requested, all 33 GiB are on 2 MiB pages; the CPU's time per
  expert stays within 1% and the start is a second faster.
- **A locked GPU clock** (`nvidia-smi -lgc`). On `struct`, six runs each: 291.6 tok/s at 2150 MHz and 283.0 at
  2000 MHz against 289.7 with the driver's own control. Under sustained heat the clock falls below the lock either
  way.
- **The pool on the efficient cores only.** Same speed, 3.7 W more package power.
- **Fewer pool workers.** 5, 7 and 11 lose to 15, although the GPU clocks 5 to 9% higher with 5.
- **Exact speculative sampling for the MTP drafter**, estimated from `--exp spec_log=1`: 78.1% of drafts kept
  today, 78.9 to 79.5% with drawn drafts.
- **Predicting the experts a layer routes next**, to bring them to the GPU in time. A table of which experts follow
  which, learned as the text goes, finds the experts that were not routed lately no better than their frequency
  does; a token's experts one layer up find 19% of them among three candidates.
- **Two requests at once** (`--parallel 2`): 187 to 199 tok/s in all against 175 to 186 one after the other, each of
  the two at about 100.

## Credits

- [Basalt](https://github.com/jesdga95/basalt) (MIT): the engine this repository forks. Its own credits are in
  [its README](https://github.com/jesdga95/basalt#credits) and in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
- [Strata](https://github.com/Niko1221/Strata) (MIT): the project Basalt forks.
- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and the other model authors: the models. Their packs carry
  the models' own licences.

## License

MIT, see [LICENSE](LICENSE).
