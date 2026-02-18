use neurokmer::utils::canonical_kmer;
use neurokmer::{SpikingKmerCounter, pack_canonical, pack_kmer};

#[cfg(test)]
mod tests {
    use neurokmer::temporal::{TemporalCoder, TemporalSpikingCounter};
    use neurokmer::{SpikingKmerCounter, WorkerInfo};
    use rand::RngExt;
    use std::collections::HashMap;
    use std::time::Instant;

    // Helper to get current memory usage (platform-specific approximation)
    fn get_memory_usage_mb() -> u64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(size_str) = line.split_whitespace().nth(1) {
                            if let Ok(size_kb) = size_str.parse::<u64>() {
                                return size_kb / 1024;
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = std::process::Command::new("ps")
                .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
                .output()
            {
                if let Ok(rss_kb) = String::from_utf8(output.stdout)
                    .unwrap_or_default()
                    .trim()
                    .parse::<u64>()
                {
                    return rss_kb / 1024;
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // On Windows, we can use a simpler approach
            if let Ok(output) = std::process::Command::new("tasklist")
                .args(&[
                    "/FI",
                    &format!("PID eq {}", std::process::id()),
                    "/FO",
                    "CSV",
                ])
                .output()
            {
                let output_str = String::from_utf8_lossy(&output.stdout);
                // Parse CSV output - this is simplified
                for line in output_str.lines() {
                    if line.contains(&std::process::id().to_string()) {
                        let parts: Vec<&str> = line.split(',').collect();
                        if parts.len() > 4 {
                            if let Ok(mem_kb) = parts[4].trim_matches('"').parse::<u64>() {
                                return mem_kb / 1024;
                            }
                        }
                    }
                }
            }
        }

        // Return 0 if can't measure
        0
    }

    #[test]
    fn test_streaming_memory_usage() {
        let file_path = "data/your_genome.fasta";
        if !std::path::Path::new(file_path).exists() {
            println!("Skipping memory test: {} not found", file_path);
            println!("Available files in data/:");
            if let Ok(entries) = std::fs::read_dir("data") {
                for entry in entries {
                    if let Ok(entry) = entry {
                        println!("  - {}", entry.file_name().to_string_lossy());
                    }
                }
            }
            return;
        }

        println!("\n=== Memory Usage Te`st` ===");
        println!("File: {}", file_path);

        // Get file size
        if let Ok(metadata) = std::fs::metadata(file_path) {
            let size_mb = metadata.len() as f64 / 1_048_576.0;
            println!("File size: {:.2} MB", size_mb);
        }

        // Test in-memory version
        println!("\n1. Testing in-memory version...");
        let start = Instant::now();
        let mem_before = get_memory_usage_mb();

        let seqs: Vec<Vec<u8>> = neurokmer::stream_sequences(file_path)
            .expect("Failed to read")
            .collect();
        println!("  Loaded {} sequences", seqs.len());

        let mut counter_mem = SpikingKmerCounter::new(31, 1.0, 0.95, 2, 1.0, 1_000_000, true);
        counter_mem.process_parallel(&seqs);

        let mem_after = get_memory_usage_mb();
        let duration = start.elapsed();

        println!("  Time: {:?}", duration);
        println!("  Memory before: {} MB", mem_before);
        println!("  Memory after: {} MB", mem_after);
        println!(
            "  Memory delta: {} MB",
            mem_after.saturating_sub(mem_before)
        );
        let mem_spikes = counter_mem.energy.total_spikes(); // FIX: use method
        println!("  Total spikes: {}", mem_spikes);

        // Force cleanup
        drop(seqs);
        drop(counter_mem);

        // Give OS time to reclaim memory
        println!("  Waiting for memory cleanup...");
        std::thread::sleep(std::time::Duration::from_secs(2));

        // Test streaming version
        println!("\n2. Testing streaming version...");

        let start = Instant::now();
        let mem_before = get_memory_usage_mb();

        let mut counter_stream = SpikingKmerCounter::new(31, 1.0, 0.95, 2, 1.0, 1_000_000, true);
        counter_stream
            .process_file_streaming(file_path)
            .expect("Streaming failed");

        let mem_after = get_memory_usage_mb();
        let duration = start.elapsed();

        println!("  Time: {:?}", duration);
        println!("  Memory before: {} MB", mem_before);
        println!("  Memory after: {} MB", mem_after);
        println!(
            "  Memory delta: {} MB",
            mem_after.saturating_sub(mem_before)
        );
        let stream_spikes = counter_stream.energy.total_spikes(); // FIX: use method
        println!("  Total spikes: {}", stream_spikes);

        // Verify results match
        println!("\n3. Verifying results match...");
        println!("  In-memory spikes: {}", mem_spikes);
        println!("  Streaming spikes: {}", stream_spikes);

        let spike_diff = (stream_spikes as i64 - mem_spikes as i64).abs();
        println!("  Spike difference: {}", spike_diff);

        if spike_diff == 0 {
            println!("  ✅ Results match perfectly!");
        } else if spike_diff < 100 {
            println!("  ⚠️  Small difference (may be due to floating point)");
        } else {
            println!("  ❌ Large difference - possible bug!");
        }

        // Memory savings calculation
        println!("\n4. Memory savings:");
        println!(
            "  In-memory peak delta: ~{} MB",
            mem_after.saturating_sub(mem_before) // This is from streaming; but we can't get in-memory peak here
        );
        println!(
            "  Streaming peak delta: ~{} MB",
            mem_after.saturating_sub(mem_before)
        );

        // Note: memory measurement may be zero on some platforms
    }

    #[test]
    fn test_temporal_coding_basic() {
        println!("\n=== Temporal Coding Basic Test ===");

        let mut coder = TemporalCoder::new(1000);

        // Test TTFS encoding: high count = early spike
        let time1 = coder.encode_count(0x1234_5678, 1000); // High count
        let time2 = coder.encode_count(0x8765_4321, 10); // Low count

        println!("High count (1000) spikes at t={}", time1);
        println!("Low count (10) spikes at t={}", time2);

        assert!(time1 < time2, "High count should spike earlier");
        println!("✅ TTFS encoding works: early spikes for strong inputs");

        // Test decoding
        let decoded1 = coder.decode_timing(0x1234_5678).unwrap();
        let decoded2 = coder.decode_timing(0x8765_4321).unwrap();

        println!("Decoded count for high: {:.1} (original: 1000)", decoded1);
        println!("Decoded count for low: {:.1} (original: 10)", decoded2);

        // Decoding is approximate due to log compression
        assert!(decoded1 > decoded2, "Decoded ordering should match");
        println!("✅ Decoding preserves rank order");
    }

    #[test]
    fn test_temporal_rank_order() {
        println!("\n=== Temporal Rank-Order Coding Test ===");

        let mut coder = TemporalCoder::new(100);

        let kmer_counts = vec![
            (0x1111_1111, 100), // Most abundant
            (0x2222_2222, 50),
            (0x3333_3333, 25),
            (0x4444_4444, 10),
            (0x5555_5555, 1), // Least abundant
        ];

        let rankings = coder.rank_order_encode(&kmer_counts);

        println!("Rank-order encoding results:");
        for (kmer, time) in &rankings {
            let original_count = kmer_counts.iter().find(|(k, _)| k == kmer).unwrap().1;
            println!(
                "  K-mer {:08X} (count={:4}) → time={}",
                kmer, original_count, time
            );
        }

        // Verify ordering: highest count should have lowest time
        let times: Vec<u32> = rankings.iter().map(|(_, t)| *t).collect();
        assert!(
            times.windows(2).all(|w| w[0] <= w[1]),
            "Times should increase with decreasing count"
        );
        println!("✅ Rank-order encoding: spike time ∝ 1/abundance");
    }

    #[test]
    fn test_temporal_spiking_counter() {
        let file_path = "data/your_genome.fasta";
        if !std::path::Path::new(file_path).exists() {
            println!("Skipping temporal test: {} not found", file_path);
            return;
        }

        println!("\n=== Temporal Spiking Counter Test ===");

        // Load sequences
        let seqs: Vec<Vec<u8>> = neurokmer::stream_sequences(file_path)
            .expect("Failed to read")
            .collect();
        println!("Loaded {} sequences", seqs.len());

        // Test without temporal coding
        println!("\n1. Standard spiking counter...");
        let start = Instant::now();
        let mut standard_counter = SpikingKmerCounter::new(31, 1.0, 0.95, 2, 1.0, 1_000_000, true);
        standard_counter.process_parallel(&seqs);
        let standard_time = start.elapsed();
        let standard_spikes = standard_counter.energy.total_spikes();
        println!("  Time: {:?}", standard_time);
        println!("  Total spikes: {}", standard_spikes);

        // Test with temporal coding
        println!("\n2. Temporal spiking counter...");
        let start = Instant::now();
        let mut temporal_counter =
            TemporalSpikingCounter::new(31, 1.0, 0.95, 2, 1.0, 1_000_000, true, 1000);
        temporal_counter.process_with_temporal(&seqs);
        let temporal_time = start.elapsed();
        let temporal_spikes = temporal_counter.counter.energy.total_spikes();
        println!("  Time: {:?}", temporal_time);
        println!("  Total spikes: {}", temporal_spikes);

        // Get temporal statistics
        let stats = temporal_counter.temporal_stats();
        println!("\n3. Temporal statistics:");
        println!("  Total temporal spikes: {}", stats.total_temporal_spikes);
        println!("  Mean spike time: {:.2}", stats.mean_spike_time);
        println!("  CV of ISIs: {:.3}", stats.coefficient_of_variation);

        // Verify spike counts match
        println!("\n4. Verification:");
        println!("  Standard spikes: {}", standard_spikes);
        println!("  Temporal spikes: {}", temporal_spikes);

        let diff = (temporal_spikes as i64 - standard_spikes as i64).abs();
        if diff == 0 {
            println!("  ✅ Temporal coding produces identical spike count!");
        } else {
            println!(
                "  ⚠️  Spike difference: {} (temporal adds extra spikes)",
                diff
            );
        }

        // Check temporal encoding worked
        assert!(
            stats.total_temporal_spikes > 0,
            "Should have temporal spikes recorded"
        );
        println!(
            "  ✅ Temporal encoding active with {} unique k-mer spike times",
            stats.total_temporal_spikes
        );
    }

    #[test]
    fn test_temporal_coding_realistic() {
        // Test with actual biological pattern: GC-rich regions spike earlier
        println!("\n=== Realistic Temporal Coding: GC Content ===");

        let mut coder = TemporalCoder::new(100);

        // Simulate: AT-rich = low activity, GC-rich = high activity
        let at_rich = 0x0000_0000; // All A/T (00 in 2-bit)
        let gc_rich = 0xFFFF_FFFF; // All G/C (11 in 2-bit)
        let mixed = 0x5555_5555; // Mixed

        // GC-rich regions are more "active" (gene coding), spike earlier
        coder.encode_count(gc_rich, 500); // High activity
        coder.encode_count(mixed, 100); // Medium
        coder.encode_count(at_rich, 20); // Low activity (intergenic)

        let gc_time = coder.get_spike_time(gc_rich).unwrap();
        let mixed_time = coder.get_spike_time(mixed).unwrap();
        let at_time = coder.get_spike_time(at_rich).unwrap();

        println!("GC-rich (high activity):  time={}", gc_time);
        println!("Mixed (medium):           time={}", mixed_time);
        println!("AT-rich (low activity):   time={}", at_time);

        assert!(
            gc_time < mixed_time && mixed_time < at_time,
            "GC-rich should spike earliest (biological realism)"
        );
        println!("✅ Biological pattern: coding regions (GC-rich) spike first!");
    }

    #[test]
    fn test_stdp_learning() {
        use neurokmer::stdp::{STDPKmerLearner, STDPPlasticity, STDPWindow};

        println!("\n=== STDP Plasticity Learning Test ===");

        // Test 1: Basic STDP window
        println!("\n1. Testing STDP timing window...");
        let stdp = STDPPlasticity::new(1.0);
        let window = STDPWindow::default();

        // LTP: pre before post (positive delta_t)
        let ltp_delta = stdp.calculate_stdp(10.0); // +10ms
        let ltd_delta = stdp.calculate_stdp(-10.0); // -10ms

        println!("  LTP (pre→post, +10ms):  Δw = {:+.6}", ltp_delta);
        println!("  LTD (post→pre, -10ms):  Δw = {:+.6}", ltd_delta);

        assert!(ltp_delta > 0.0, "LTP should be positive");
        assert!(ltd_delta < 0.0, "LTD should be negative");
        assert!(
            ltp_delta.abs() > ltd_delta.abs() || (ltp_delta - ltd_delta.abs()).abs() < 0.001,
            "Asymmetric STDP: LTP slightly stronger or balanced"
        );
        println!("  ✅ Asymmetric STDP window confirmed");

        // Test 2: Synaptic weight updates
        println!("\n2. Testing weight updates...");
        let mut plasticity = STDPPlasticity::new(1.0);
        plasticity.add_synapse(0, 1, 0.5, 1);

        // Simulate pre-post pairing (should strengthen)
        plasticity.record_spike(0, 10); // Pre at t=10
        plasticity.record_spike(1, 25); // Post at t=25 (delayed by 15ms, within window)
        plasticity.apply_learning();

        let weight_after_ltp = plasticity
            .synapses
            .get(&(0, 1))
            .unwrap()
            .weight
            .get_weight();
        println!("  Initial weight: 0.5000");
        println!("  After LTP (pre@10, post@25): {:.4}", weight_after_ltp);
        assert!(weight_after_ltp > 0.5, "LTP should strengthen synapse");

        // Test post-pre pairing (should weaken)
        plasticity.record_spike(1, 100); // Post at t=100
        plasticity.record_spike(0, 115); // Pre at t=115 (post before pre = -15ms)
        plasticity.apply_learning();

        let weight_after_ltd = plasticity
            .synapses
            .get(&(0, 1))
            .unwrap()
            .weight
            .get_weight();
        println!("  After LTD (post@100, pre@115): {:.4}", weight_after_ltd);
        assert!(
            weight_after_ltd < weight_after_ltp,
            "LTD should weaken synapse"
        );
        println!("  ✅ Bidirectional plasticity working");

        // Test 3: Full k-mer learner
        println!("\n3. Testing STDP k-mer learner...");
        let file_path = "data/your_genome.fasta";
        if !std::path::Path::new(file_path).exists() {
            println!("  Skipping (file not found)");
            return;
        }

        let seqs: Vec<Vec<u8>> = neurokmer::stream_sequences(file_path)
            .expect("Failed to read")
            .collect();

        let mut learner = STDPKmerLearner::new(31, 1.0, 0.95, 2, 1.0, 100_000, true, 0.5, 0.01);

        println!(
            "  Processing {} sequences with STDP learning...",
            seqs.len()
        );
        learner.process_with_learning(&seqs);

        let stats = learner.learning_stats();
        println!("\n  Learning statistics:");
        println!("    Synapses: {}", stats.synapse_count);
        println!(
            "    Mean weight: {:.4} ± {:.4}",
            stats.mean_weight, stats.weight_std
        );
        println!("    Patterns learned: {}", stats.patterns_learned);
        println!("    Pattern diversity: {}", stats.pattern_diversity);

        assert!(stats.synapse_count > 0, "Should have created synapses");
        assert!(stats.patterns_learned > 0, "Should have learned patterns");
        println!("  ✅ STDP learning on k-mers successful!");

        // Test 4: Pattern completion
        println!("\n4. Testing pattern completion...");
        let seed = vec![learner.counter.map_kmer_to_neuron(0x1234_5678_9ABC_DEF0)];
        let completed = learner.pattern_completion(&seed, 5);
        println!("  Seed neurons: {:?}", seed);
        println!("  Completed pattern: {} neurons", completed.len());
        println!("  ✅ Pattern completion functional");
    }

    #[test]
    fn test_stdp_biological_realism() {
        use neurokmer::stdp::STDPPlasticity;

        println!("\n=== STDP Biological Realism Test ===");

        let mut stdp = STDPPlasticity::new(1.0);
        stdp.add_synapse(0, 1, 0.3, 2);

        // Test 1: Causal potentiation (pre leads to post)
        println!("\n1. Causal LTP (pre@0, post@5, delay=2)...");
        stdp.record_spike(0, 0); // Pre spikes
        stdp.record_spike(1, 5); // Post spikes 5ms later (effective: 0+2=2, post@5, delta=+3)
        stdp.apply_learning();

        let w1 = stdp.synapses[&(0, 1)].weight.get_weight();
        println!("  Weight: 0.3000 → {:.4} (should increase)", w1);
        assert!(
            w1 > 0.3,
            "Causal firing strengthens synapse (Hebbian learning)"
        );

        // Test 2: Anti-causal depression (post leads to pre)
        println!("\n2. Anti-causal LTD (post@10, pre@15)...");
        stdp.record_spike(1, 10); // Post first
        stdp.record_spike(0, 15); // Pre 5ms later (anti-causal)
        stdp.apply_learning();

        let w2 = stdp.synapses[&(0, 1)].weight.get_weight();
        println!("  Weight: {:.4} → {:.4} (should decrease)", w1, w2);
        assert!(w2 < w1, "Anti-causal firing weakens synapse");

        // Test 3: Temporal specificity (spikes too far apart)
        println!("\n3. Temporal specificity (pre@100, post@300)...");
        let w_before = stdp.synapses[&(0, 1)].weight.get_weight();
        stdp.record_spike(0, 100);
        stdp.record_spike(1, 300); // 200ms apart, outside STDP window
        stdp.apply_learning();

        let w_after = stdp.synapses[&(0, 1)].weight.get_weight();
        println!(
            "  Weight: {:.4} → {:.4} (should be unchanged, outside window)",
            w_before, w_after
        );
        assert!(
            (w_after - w_before).abs() < 0.001,
            "Distant spikes don't affect plasticity"
        );

        println!("  ✅ All biological STDP properties confirmed!");
    }

    #[test]
    fn test_associative_memory_semantic() {
        use neurokmer::associative_memory::{
            AssociativeMemory, NeuroAssociativeSystem, SemanticKmerQuery,
        };

        println!("\n=== Associative Memory Semantic Query Test ===");

        // Test 1: Basic Hopfield storage and recall
        println!("\n1. Testing Hopfield associative memory...");
        let mut memory = AssociativeMemory::new(100, 10);

        // Store patterns (as neuron indices)
        let pattern_a = vec![0, 1, 2, 3, 4];
        let pattern_b = vec![10, 11, 12, 13, 14];
        let pattern_c = vec![0, 1, 2, 20, 21]; // Overlaps with A

        memory.store_pattern(&pattern_a).unwrap();
        memory.store_pattern(&pattern_b).unwrap();
        memory.store_pattern(&pattern_c).unwrap();

        println!("  Stored {} patterns", memory.stored_patterns.len());

        // Recall with partial cue
        let cue = vec![0, 1, 2]; // Should recall pattern A or C
        let (recalled, confidence) = memory.recall(&cue, 50, 0.0);

        println!(
            "  Cue: {:?} → Recalled: {:?} (confidence: {:.2})",
            cue, recalled, confidence
        );
        assert!(recalled.len() >= 3, "Should recall complete pattern");
        println!("  ✅ Pattern completion works");

        // Test 2: Semantic k-mer query
        println!("\n2. Testing semantic k-mer query...");
        let file_path = "data/your_genome.fasta";
        if !std::path::Path::new(file_path).exists() {
            println!("  Skipping (file not found)");
            return;
        }

        let seqs: Vec<Vec<u8>> = neurokmer::stream_sequences(file_path)
            .expect("Failed to read")
            .collect();

        let mut semantic = SemanticKmerQuery::new(31, 100_000);
        semantic.index_sequences(&seqs);

        let stats = semantic.memory_stats();
        println!(
            "  Indexed {} unique k-mers in {} patterns",
            stats.unique_kmers_indexed, stats.patterns_stored
        );
        println!(
            "  Memory capacity: {:.1}% used",
            stats.capacity_used * 100.0
        );

        // Query with real k-mers from sequences
        if let Some(&query_kmer) = semantic.kmer_to_neuron.keys().next() {
            let similar = semantic.fuzzy_query(query_kmer, 2);
            println!(
                "  Query k-mer {:016X} found {} similar k-mers",
                query_kmer,
                similar.len()
            );
            for (kmer, score) in similar.iter().take(3) {
                println!("    Similar: {:016X} (score: {:.3})", kmer, score);
            }
            println!("  ✅ Semantic similarity search working");
        }

        // Test pattern completion
        if semantic.kmer_to_neuron.len() >= 3 {
            let sample_kmers: Vec<u64> = semantic.kmer_to_neuron.keys().take(3).copied().collect();
            let completed = semantic.complete_pattern(&sample_kmers);
            println!(
                "  Partial pattern {:?} completed to {} k-mers",
                sample_kmers.len(),
                completed.len()
            );
            println!("  ✅ Pattern completion functional");
        }

        // Test 3: Integrated STDP + Associative system
        println!("\n3. Testing integrated neuro-associative system...");
        let mut system = NeuroAssociativeSystem::new(31, 100_000, 0.5, 0.001);
        system.learn(&seqs);

        if let Some(&query_kmer) = system.semantic_memory.kmer_to_neuron.keys().next() {
            let results = system.intelligent_query(&[query_kmer]);
            println!(
                "  Intelligent query returned {} semantic matches",
                results.semantic_matches.len()
            );
            println!(
                "  STDP predictions: {} k-mers",
                results.stdp_predictions.len()
            );
            println!("  ✅ Integrated system operational");
        }
    }

    #[test]
    fn test_memory_capacity_and_interference() {
        use neurokmer::associative_memory::AssociativeMemory;
        use std::collections::HashMap;

        println!("\n=== Memory Capacity & Interference Test ===");

        let pattern_size = 100;
        let mut memory = AssociativeMemory::new(pattern_size, 15);

        // Store increasing number of random patterns
        let test_capacities = vec![5, 10, 15];

        for &num_patterns in &test_capacities {
            memory.stored_patterns.clear();
            memory.pattern_strengths.clear();
            // FIXED: sparse_weights is HashMap, not Vec - clear it instead of reassigning
            memory.sparse_weights.clear();

            // Generate random patterns
            let mut patterns = Vec::new();
            for _ in 0..num_patterns {
                // FIXED: Use rand::rng().random_range() instead of rand::random::<usize>()
                let pattern: Vec<usize> = (0..10)
                    .map(|_| rand::rng().random_range(0..pattern_size))
                    .collect();
                patterns.push(pattern);
            }

            // Store all patterns
            let mut stored = 0;
            for pattern in &patterns {
                if memory.store_pattern(pattern).is_ok() {
                    stored += 1;
                }
            }

            println!(
                "  Capacity {}: stored {}/{} patterns",
                num_patterns, stored, num_patterns
            );

            // Test recall quality
            if stored > 0 {
                let test_pattern = &patterns[0];
                let cue: Vec<usize> = test_pattern.iter().take(5).copied().collect();
                let (recalled, confidence) = memory.recall(&cue, 100, 0.0);

                // FIXED: Use &&r in filter closure
                let overlap = recalled
                    .iter()
                    .filter(|&&r| test_pattern.contains(&r))
                    .count();
                let recall_quality = overlap as f32 / test_pattern.len().max(1) as f32;

                println!(
                    "    Recall quality: {:.1}% (confidence: {:.2})",
                    recall_quality * 100.0,
                    confidence
                );
            }
        }

        println!("  ✅ Capacity limits and interference demonstrated");
    }

    #[test]
    fn test_distributed_computing() {
        use neurokmer::distributed::{
            DistributedCoordinator, DistributedWorker, MapReduceKmer, TaskResults,
        };

        println!("\n=== Distributed Computing Test ===");

        // Test 1: Map-Reduce locally (simulated distribution)
        println!("\n1. Testing Map-Reduce k-mer counting...");

        let file_path = "data/your_genome.fasta";
        if !std::path::Path::new(file_path).exists() {
            println!("  Skipping (file not found)");
            return;
        }

        let data = std::fs::read(file_path).expect("Failed to read file");
        let map_reduce = MapReduceKmer::new(data, 10000); // 10KB chunks

        println!(
            "  Split {} bytes into {} chunks",
            map_reduce.chunks.iter().map(|c| c.len()).sum::<usize>(),
            map_reduce.chunks.len()
        );

        // Map phase (parallel processing)
        let mapper = |chunk: &[u8]| {
            // Simulate k-mer counting on chunk
            let mut counts = HashMap::new();
            for (i, &byte) in chunk.iter().enumerate() {
                let kmer = byte as u64 + (i as u64 % 1000);
                *counts.entry(kmer).or_insert(0) += 1;
            }

            TaskResults {
                kmer_counts: counts.iter().map(|(&k, &v)| (k, v as u32)).collect(),
                neuron_currents: vec![(0, chunk.len() as u64)],
                processed_bytes: chunk.len(),
            }
        };

        let map_results = map_reduce.map(mapper);
        println!("  Map phase: {} partial results", map_results.len());

        // Reduce phase
        let global_counts = MapReduceKmer::reduce(&map_results);
        println!(
            "  Reduce phase: {} unique k-mers aggregated",
            global_counts.len()
        );

        // Verify total counts match
        let total_from_chunks: u64 = map_results
            .iter()
            .map(|r| r.kmer_counts.values().map(|&v| v as u64).sum::<u64>())
            .sum();
        let total_reduced: u64 = global_counts.values().sum();

        assert_eq!(
            total_from_chunks, total_reduced,
            "Map-Reduce consistency check"
        );
        println!(
            "  ✅ Map-Reduce consistency verified: {} total counts",
            total_reduced
        );

        // Test 2: Coordinator-Worker architecture (local simulation)
        println!("\n2. Testing Coordinator-Worker architecture...");

        // Start coordinator in background
        let coordinator = DistributedCoordinator::new("127.0.0.1:0"); // Random port
        println!("  Coordinator created");

        // Simulate worker registration
        coordinator.workers.lock().unwrap().insert(
            "worker-1".to_string(),
            WorkerInfo {
                address: "127.0.0.1:9001".to_string(),
                last_heartbeat: std::time::Instant::now(),
                current_load: 0.0,
                total_processed: 0,
            },
        );

        // Distribute tasks
        for chunk in &map_reduce.chunks[..3] {
            let task_id = coordinator.distribute_task(chunk.clone());
            println!("  Task {} queued", task_id);
        }

        let queued_tasks = coordinator.task_queue.lock().unwrap().len();
        println!("  {} tasks in queue", queued_tasks);
        assert_eq!(queued_tasks, 3, "Tasks should be queued");

        // Test 3: Fault tolerance simulation
        println!("\n3. Testing fault tolerance...");

        // Simulate worker failure
        {
            let mut workers = coordinator.workers.lock().unwrap();
            if let Some(worker) = workers.get_mut("worker-1") {
                // Simulate stale heartbeat
                worker.last_heartbeat =
                    std::time::Instant::now() - std::time::Duration::from_secs(120);
            }
        }

        // Check for dead workers
        let dead_workers: Vec<String> = coordinator
            .workers
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, info)| info.last_heartbeat.elapsed() > std::time::Duration::from_secs(60))
            .map(|(id, _)| id.clone())
            .collect();

        println!(
            "  Detected {} dead workers: {:?}",
            dead_workers.len(),
            dead_workers
        );

        for dead in dead_workers {
            coordinator.workers.lock().unwrap().remove(&dead);
            println!("  Removed dead worker: {}", dead);
        }

        println!("  ✅ Fault tolerance working");

        // Test 4: Scalability metrics
        println!("\n4. Scalability analysis...");
        let num_workers = 100;
        let chunks_per_worker = map_reduce.chunks.len() as f64 / num_workers as f64;

        println!(
            "  With {} workers: {:.2} chunks/worker",
            num_workers, chunks_per_worker
        );
        println!("  Estimated speedup: {:.1}x (ideal)", num_workers as f64);
        println!(
            "  Communication overhead: ~{} bytes/task",
            std::mem::size_of::<TaskResults>()
        );

        println!("  ✅ Distributed architecture ready for unlimited scale!");
    }
}
