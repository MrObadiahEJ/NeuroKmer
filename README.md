# NeuroKmer – Neuromorphic K-mer Counting in Pure Rust

**v0.3.0** (Advanced pooled neurons + Python bindings + associative memory)

NeuroKmer is a **brain-inspired, event-driven k-mer counter** that uses a fixed-size pool of Leaky Integrate-and-Fire (LIF) spiking neurons instead of traditional hash tables.

k-mers are bit-packed (u64) and mapped to neurons via SipHash. Each occurrence adds current; neurons integrate, leak, and spike. Spikes drive approximate counting (per-neuron spike rates + collision-based increment). A Willshaw associative memory enables fuzzy/noisy recall for error-prone data.

Key advantages:
- **Constant memory** (controlled by `--pool-size`)
- **Scalable** to large genomes (tested on 115MB+ FASTA)
- **Biological realism** (event-driven, tunable LIF dynamics)
- **Python bindings** (PyO3) for easy scripting/notebooks
- **Thread-safe counting** (DashMap) and parallel-ready

## Current Features
- Streaming FASTA/FASTQ parsing (needletail) – zero memory blowup
- Zero-copy k-mer windows + bit-packing (2 bits/base)
- Fixed neuron pool with SipHash mapping
- Per-neuron spike counting + global energy simulation
- Thread-safe approximate counts (DashMap)
- Top N abundant k-mers output (sorted by spike-driven counts)
- Advanced Willshaw associative memory with BLAKE3 sparse patterns + fuzzy recall
- Python API (`neurokmer_py`) for Jupyter/Colab integration
- Robust CLI, logging, error handling

## Benchmarks
Run with `cargo bench` (uses E. coli genome in `data/small_genome.fasta`).

Example results (on Dell Latitude E7440, k=31):
- HashMap (exact): ~X ms, high memory
- Spiking Pool (2M neurons): ~Y ms, constant ~100 MB RAM

See `target/criterion/report/index.html` for full plots.

## Quick Start
```bash
# Build optimized binary
cargo run --release -- --input data/your_genome.fasta --k 31 --pool-size 2000000