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

    #[arg(long, default_value_t = 1_000_000)]
    pool_size: usize,

    #[arg(long, default_value_t = false)]
    canonical: bool,

    #[arg(long, default_value_t = false)]
    streaming: bool, // New flag for streaming mode
}

fn main() -> NeuroResult<()> {
    let args = Cli::parse();
    init_logging()?;

    info!(
        "Starting NeuroKmer on {} (k={}, pool_size={}, canonical={}, streaming={})",
        args.input, args.k, args.pool_size, args.canonical, args.streaming
    );

    let mut counter =
        SpikingKmerCounter::new(args.k, 1.0, 0.95, 2, 1.0, args.pool_size, args.canonical);

    if args.streaming {
        // Streaming mode: memory efficient, processes file directly
        counter.process_file_streaming(&args.input)?;
    } else {
        // Original mode: loads all into memory first
        let seqs: Vec<Vec<u8>> = stream_sequences(&args.input)?.collect();
        counter.process_parallel(&seqs);
    }

    // Output results (same as before)
    println!("\n=== Top 20 Abundant Neuron Groups (Highest Spike Rates) ===");
    let top = counter.top_abundant_neurons(20);
    if top.is_empty() {
        println!("No spikes fired (empty file or too small k)");
    } else {
        for (rank, (idx, spikes, uniques)) in top.iter().enumerate() {
            println!(
                "{:3}: Neuron {:6} → {:8} spikes ({} unique k-mers colliding)",
                rank + 1,
                idx,
                spikes,
                uniques
            );
        }
    }

    info!(
        "Processing complete – total spikes: {}, energy: {}",
        counter.energy.total_spikes(),
        counter.energy_used()
    );

    println!("\nTotal spikes fired: {}", counter.energy.total_spikes());
    println!("Simulated energy used: {}", counter.energy_used());
    println!("Neuron pool size used: {}", args.pool_size);
    println!("Streaming mode: {}", args.streaming);

    Ok(())
}
