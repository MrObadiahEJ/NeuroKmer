use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use neurokmer::{SpikingKmerCounter, stream_sequences};
use std::time::Duration;

fn bench_with_real_files(c: &mut Criterion) {
    let files = [
        ("sm.fasta", "0.8KB"),
        ("md.fasta", "1.5MB"),
        ("sample1.fasta", "115MB"),
    ];

    let k = 31;
    let pool_size = 1_000_000;

    for (file_name, size) in files.iter() {
        let file_path = format!("data/{}", file_name);

        // Skip if file doesn't exist
        if !std::path::Path::new(&file_path).exists() {
            eprintln!("Warning: {} not found, skipping", file_path);
            continue;
        }

        let mut group = c.benchmark_group(format!("File: {} ({})", file_name, size));

        // Configure for longer benchmarks with large files
        if *file_name == "sample1.fasta" {
            group.sample_size(10); // Fewer samples for large file
            group.measurement_time(Duration::from_secs(30));
            group.warm_up_time(Duration::from_secs(5));
        }

        // Benchmark in-memory version
        group.bench_function("In-memory", |b| {
            b.iter(|| {
                let seqs: Vec<Vec<u8>> = stream_sequences(&file_path)
                    .expect("Failed to read")
                    .collect();
                let mut counter = SpikingKmerCounter::new(k, 1.0, 0.95, 2, 1.0, pool_size, true);
                counter.process_parallel(&seqs);
                black_box(counter.energy.total_spikes);
            })
        });

        // Benchmark streaming version
        group.bench_function("Streaming", |b| {
            b.iter(|| {
                let mut counter = SpikingKmerCounter::new(k, 1.0, 0.95, 2, 1.0, pool_size, true);
                counter
                    .process_file_streaming(&file_path)
                    .expect("Streaming failed");
                black_box(counter.energy.total_spikes);
            })
        });

        group.finish();
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10))
        .sample_size(50);
    targets = bench_with_real_files
);
criterion_main!(benches);
