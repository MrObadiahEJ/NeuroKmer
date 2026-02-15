# NeuroKmer – Neuromorphic K-mer Counting in Pure Rust

**v0.2.2** (Scalable pooled neuron edition)

NeuroKmer is a brain-inspired k-mer counter that replaces traditional hash tables with a **fixed-size pool of Leaky Integrate-and-Fire (LIF) spiking neurons**.

Every k-mer occurrence sends a small current to a neuron (mapped via SipHash on bit-packed k-mers). Neurons integrate, leak, and spike when their voltage crosses threshold. Spikes and simulated energy serve as proxies for k-mer abundance – an **event-driven, low-power** approach inspired by biological neural populations.

Unlike exact counters (Jellyfish, KMC), NeuroKmer uses **constant memory** (set by `--pool-size`, default 1M neurons ≈ 50–100 MB) and is designed for future neuromorphic hardware or noisy long-read data.

## Current Features
- Streaming FASTA/FASTQ parsing (`needletail`) – memory-efficient for huge files
- Bit-packed k-mers (u64, 2 bits/base) – fast and compact
- Fixed-size neuron pool with SipHash mapping – constant memory, scalable to real genomes
- Per-neuron spike counting + global energy simulation
- Robust CLI (clap), logging, error handling
- Parallel-ready (rayon imported, easy to add)

## Quick Start
```bash
# Build optimized binary
cargo run --release -- --input data/your_genome.fasta --k 31 --pool-size 2000000