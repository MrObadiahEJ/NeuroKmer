// benches/compare_counters.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neurokmer::{SpikingKmerCounter, stream_sequences};
use std::collections::HashMap;

fn bench_spiking_vs_hashmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("K-mer Counting (k=31)");

    // Load sequences once
    let seqs: Vec<Vec<u8>> = stream_sequences("data/small_genome.fasta")
        .expect("Failed to read test file")
        .collect();

    group.bench_function("HashMap (exact)", |b| {
        b.iter(|| {
            let mut counts = HashMap::new();
            for seq in &seqs {
                for window in seq.windows(31) {
                    let kmer = neurokmer::utils::pack_kmer(window);
                    *counts.entry(kmer).or_insert(0) += 1;
                }
            }
            black_box(counts.len())
        })
    });

    group.bench_function("Spiking Pool (approximate)", |b| {
        b.iter(|| {
            let mut counter = SpikingKmerCounter::new(31, 1.0, 0.95, 2, 1.0, 2_000_000);
            for seq in &seqs {
                counter.process_sequence(seq);
            }
            black_box(counter.energy.total_spikes)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_spiking_vs_hashmap);
criterion_main!(benches);