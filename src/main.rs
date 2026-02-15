// src/main.rs
// Clean, compiling version for the current pooled neuron design
// No associative memory integration yet (to avoid errors – add later)
// Uses fixed pool_size for constant memory

use clap::Parser;
use log::info;
use neurokmer::{NeuroResult, SpikingKmerCounter, init_logging, stream_sequences};

#[derive(Parser, Debug)]
#[command(
    name = "neurokmer",
    about = "Neuromorphic k-mer counting with fixed-size spiking neuron pool"
)]
struct Cli {
    #[arg(short, long)]
    input: String,

    #[arg(short, long, default_value_t = 31)]
    k: usize,

    #[arg(long, default_value_t = 1.0)]
    threshold: f64,

    #[arg(long, default_value_t = 0.95)]
    leak: f64,

    #[arg(long, default_value_t = 2)]
    refractory: u32,

    #[arg(long, default_value_t = 1_000_000)]
    pool_size: usize, // Fixed neuron pool – controls memory/accuracy trade-off
}

fn main() -> NeuroResult<()> {
    let args = Cli::parse();
    init_logging()?;

    info!(
        "Starting NeuroKmer on {} (k={}, pool_size={})",
        args.input, args.k, args.pool_size
    );

    let mut counter = SpikingKmerCounter::new(
        args.k,
        args.threshold,
        args.leak,
        args.refractory,
        1.0,            // spike_cost
        args.pool_size, // Fixed pool size
    );

    for seq in stream_sequences(&args.input)? {
        counter.process_sequence(&seq);
    }
    println!("\n=== K-mer Counts (first 20) ===");
    let mut counts: Vec<_> = counter.counts.iter().map(|ref_multi| (*ref_multi.key(), *ref_multi.value())).collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by count descending

    for (i, (kmer, count)) in counts.iter().take(20).enumerate() {
        println!("{:3}: k-mer {:016x} → {} counts", i + 1, kmer, count);
    }

    println!("\nTotal distinct k-mers: {}", counts.len());
    info!(
        "Processing complete – total spikes: {}, energy: {}",
        counter.energy.total_spikes,
        counter.energy_used()
    );

    println!("Total spikes fired: {}", counter.energy.total_spikes);
    println!("Simulated energy used: {}", counter.energy_used());
    println!("Neuron pool size used: {}", args.pool_size);

    Ok(())
}
