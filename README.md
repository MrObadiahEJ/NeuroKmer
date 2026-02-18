````markdown
# NeuroKmer – Neuromorphic K-mer Counting in Pure Rust

**v0.6.0** (SIMD acceleration + f32 LIF neurons + full neuromorphic stack)

NeuroKmer is a **brain-inspired, event-driven k-mer counter** that replaces traditional hash tables with a fixed-size pool of Leaky Integrate-and-Fire (LIF) spiking neurons.

k-mers are processed with **canonical representation** (lexicographically smaller of forward/reverse complement) using an efficient **rolling hash** (O(1) per k-mer). Each occurrence adds current to a pooled neuron (SipHash mapping on bit-packed u64 k-mers). Neurons integrate, leak, and spike—spike rates provide accurate abundance approximation.

**v0.6.0 highlights**:

- Full **SIMD (AVX2) acceleration** on x86_64 → ~1.5× faster spike simulation
- f32 LIF neurons for vectorized computation
- Perfect spike match between scalar, parallel, and SIMD modes
- Temporal coding, STDP plasticity, associative semantic queries, and distributed architecture all integrated

Key advantages:

- **Constant memory** (set by `--pool-size`, e.g., 1M neurons ≈ 50–100 MB)
- **True low-memory streaming** (parallel workers + channels, O(threads × max_seq_len))
- **High performance** — SIMD + rayon parallelism
- **Biological realism** — tunable LIF, temporal coding, STDP learning, associative recall
- **Real-world tested** — perfect match on 115 MB+ FASTA files across all modes

## Current Features

- Streaming FASTA/FASTQ parsing (`needletail`)
- Canonical k-mers with rolling hash (O(1) sliding window)
- Fixed neuron pool with SipHash mapping
- Per-neuron spike counting + global energy simulation
- Thread-safe approximate counts (DashMap + atomics)
- **SIMD-accelerated LIF simulation** (AVX2 8-wide batching)
- Temporal coding (TTFS + rank-order encoding)
- STDP plasticity for Hebbian learning
- Willshaw associative memory + semantic/fuzzy k-mer queries
- Distributed map-reduce architecture (coordinator + workers)
- Configurable simulation steps
- Robust CLI, logging, error handling

## Quick Start

```bash
cargo run --release -- --input data/your_genome.fasta --k 31 --pool-size 2000000 --canonical --streaming
```
````

Example output on 115 MB FASTA (k=31, 2M pool):

```
=== Top 20 Abundant Neuron Groups (Highest Spike Rates) ===
  1: Neuron 123456 →   1234567 spikes (45 unique k-mers colliding)
...
Total spikes fired: 76082638
Simulated energy used: 76082638
Neuron pool size used: 2000000
Streaming mode: true
```

## Performance & Memory Test Results

Tested on 115 MB FASTA (7 sequences):

- **In-memory mode**: ~10.3 min, 76M spikes
- **Streaming mode**: ~7.6 min, **identical 76M spikes**, constant ~100 MB peak RAM

On small genomes (0.07 MB):

- In-memory: ~4–6 s
- Streaming + SIMD: comparable or faster with AVX2 enabled

Spikes match **perfectly** (difference = 0) across all modes — verified in tests.

## Roadmap / Next Steps

All core + advanced phases are now complete!  
Next focus: real-world benchmarks on full human genomes (GRCh38 ~3 GB), paper draft, and open-source community contributions.

## Dependencies

```toml
bio = "3.0.0"
blake3 = "1.8.3"
clap = { version = "4.5.58", features = ["derive"] }
crossbeam-channel = "0.5.15"
dashmap = "6.1.0"
either = "1.15.0"
env_logger = "0.11.9"
log = "0.4.29"
ndarray = "0.17.2"
needletail = "0.6.3"
rand = "0.10.0"
rayon = "1.11.0"
siphasher = "1.0.2"

[dev-dependencies]
criterion = "0.8.2"
```

Author: Авдий Komguem (@mrobadiahej on X / LinkedIn / all socials)

```

```
