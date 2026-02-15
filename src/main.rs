use clap::Parser;
use log::info;
use neurokmer::{init_logging, stream_sequences, SpikingKmerCounter, NeuroResult};

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
    pool_size: usize,  // Fixed neuron pool – controls memory/accuracy trade-off
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
        1.0,     // threshold (fixed – tunable later)
        0.95,    // leak
        2,       // refractory
        1.0,     // spike_cost
        args.pool_size,
    );

    // Collect all sequences once (streaming but in-memory for parallel)
    let seqs: Vec<Vec<u8>> = stream_sequences(&args.input)?.collect();

    // Always use parallel version – fast & safe with atomics
    counter.process_parallel(&seqs);

    // Output top abundant neuron groups
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
        counter.energy.total_spikes,
        counter.energy_used()
    );

    println!("\nTotal spikes fired: {}", counter.energy.total_spikes);
    println!("Simulated energy used: {}", counter.energy_used());
    println!("Neuron pool size used: {}", args.pool_size);

    Ok(())
}