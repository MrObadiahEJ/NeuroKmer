```markdown
# NeuroKmer – Neuromorphic K-mer Counting in Pure Rust

**v0.4.0** (True streaming + canonical k-mers + rolling hash)

NeuroKmer is a **brain-inspired, event-driven k-mer counter** that replaces traditional hash tables with a fixed-size pool of Leaky Integrate-and-Fire (LIF) spiking neurons.

k-mers are processed with **canonical representation** (lexicographically smaller of forward/reverse complement) using an efficient **rolling hash** (O(1) per k-mer). Each occurrence adds current to a pooled neuron (SipHash mapping on bit-packed u64 k-mers). Neurons integrate, leak, and spike—spike rates provide accurate abundance approximation.

Key advantages:
- **Constant memory** (set by `--pool-size`, e.g., 1M neurons ≈ 50–100 MB)
- **True low-memory streaming** (parallel workers + channels, O(threads × max_seq_len))
- **Full parallelism** (rayon + atomics)
- **Biological realism** (tunable LIF dynamics, event-driven, energy simulation)
- **Real-world tested** on 115MB+ FASTA files with **perfect in-memory/streaming result match**

## Current Features
- Streaming FASTA/FASTQ parsing (`needletail`)
- Bit-packed + canonical k-mers with rolling hash
- Fixed neuron pool with SipHash mapping
- Per-neuron spike counting + global energy simulation
- Thread-safe approximate counts (DashMap + atomics)
- Top N abundant neuron groups output (highest spike rates = dominant k-mer groups)
- Configurable simulation steps
- Robust CLI, logging, error handling

## Quick Start
```bash
cargo run --release -- --input data/your_genome.fasta --k 31 --pool-size 2000000 --canonical --streaming
```

Example output on 115MB FASTA (k=31, 2M pool):
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
Tested on 115MB FASTA (7 sequences):
- **In-memory mode**: ~10.3 min, 76M spikes
- **Streaming mode**: ~7.6 min, **identical 76M spikes**, constant ~100MB peak RAM

Spikes match perfectly between modes (verified in tests).

## Roadmap / Next Steps
1. **SIMD + f32 LIF Neurons** – 3-6x speedup (hard)
2. **Temporal Coding** – Biological realism (medium)
3. **STDP Plasticity** – Learning capability (hard)
4. **Associative Memory Integration** – Semantic/fuzzy queries (medium)
5. **Distributed Computing** – Unlimited scale (hard)

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

Author: Авдий Komguem (@mrobadiahej on X / LinkedIn / all socials) – BTS Génie Logiciel expected 12–14/20 (August 2025), preparing Master's in Russia 2026–2027 (bioinformatics/neuromorphic computing via fully funded Russian Government Scholarship).