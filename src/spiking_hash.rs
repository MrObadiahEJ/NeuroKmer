use crate::{NeuroResult, RollingKmerHash};
// src/spiking_hash.rs
// Uses borrowed LIF for event-driven counting
// One neuron per unique k-mer (or hash-mapped for efficiency)
use crate::models::{EnergyTracker, LifNeuron};
use crossbeam_channel::unbounded;
use dashmap::DashMap;
use rayon::prelude::*;
use siphasher::sip::SipHasher13;
use std::collections::HashMap;
// Add to Cargo.toml: siphasher = "0.3"
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::thread;

pub struct SpikingKmerCounter {
    neurons: Vec<LifNeuron>, // Fixed pool (e.g., 1M neurons)
    pub energy: EnergyTracker,
    pub k: usize,
    pool_size: usize, // NEW: Fixed neuron count
    spike_cost: f64,
    threshold: f32,
    leak: f32,
    refractory: u32,
    neuron_currents: Vec<AtomicU64>, // Thread-safe atomic for parallel accumulation
    pub kmer_per_neuron: DashMap<usize, u32>, // Optional: Keep track of unique k-mers per neuron for collision stats
    pub counts: DashMap<u64, AtomicU32>,
    neuron_to_kmers: Vec<Vec<u64>>, // Track which k-mers mapped to each neuron
    pub use_canonical: bool,
    steps: usize, // Number of steps to simulate for spike generation
    // Add these for SIMD batching
    batch_voltages: Vec<f32>,
    batch_thresholds: Vec<f32>,
    batch_leaks: Vec<f32>,
    batch_refractory: Vec<u32>,
    current_multiplier: f32,
}

impl SpikingKmerCounter {
    pub fn new(
        k: usize,
        threshold: f32,
        leak: f32,
        refractory: u32,
        spike_cost: f64,
        pool_size: usize,
        use_canonical: bool,
    ) -> Self {
        let neurons: Vec<LifNeuron> = (0..pool_size)
            .map(|_| LifNeuron::new(threshold, leak, refractory))
            .collect();
        let neuron_currents: Vec<AtomicU64> = (0..pool_size).map(|_| AtomicU64::new(0)).collect();
        let neuron_to_kmers = (0..pool_size).map(|_| Vec::new()).collect();
        let padded_size = (pool_size + 7) & !7;

        Self {
            neurons,
            energy: EnergyTracker::new(),
            k,
            pool_size,
            spike_cost,
            neuron_currents,
            kmer_per_neuron: DashMap::new(),
            neuron_to_kmers,
            threshold,
            leak,
            refractory,
            counts: DashMap::new(),
            use_canonical,
            steps: 1000, // Default steps for spike simulation
            batch_voltages: vec![0.0; padded_size],
            batch_thresholds: vec![threshold as f32; padded_size],
            batch_leaks: vec![leak as f32; padded_size],
            batch_refractory: vec![0; padded_size],
            current_multiplier: 1000.0,
        }
    }
    fn map_kmer_to_neuron(&self, packed: u64) -> usize {
        let mut hasher = SipHasher13::new_with_keys(0, 0);
        packed.hash(&mut hasher);
        (hasher.finish() % self.pool_size as u64) as usize
    }

    pub fn process_parallel(&mut self, seqs: &[Vec<u8>]) {
        log::warn!(
            "process_parallel loads all sequences into memory. For large files, use process_file_streaming()"
        );

        let k = self.k;
        let pool_size = self.pool_size;
        let use_canonical = self.use_canonical;

        // Parallel fold with canonical processing
        let (total_currents, local_count_maps) = seqs
            .par_iter()
            .fold(
                || (vec![0u64; pool_size], Vec::<HashMap<u64, u32>>::new()),
                |(mut currents, mut maps), seq| {
                    let mut local_counts = HashMap::new();
                    let mut local_unique = vec![false; pool_size];

                    if use_canonical && seq.len() >= k {
                        // Use rolling hash
                        let mut rolling = RollingKmerHash::new(k);

                        // Process first k-mer
                        rolling.init(&seq[0..k]);
                        let packed = std::cmp::min(rolling.forward(), rolling.reverse_complement());

                        *local_counts.entry(packed).or_insert(0) += 1;
                        let idx = self.map_kmer_to_neuron(packed);
                        currents[idx] += 1;
                        local_unique[idx] = true;

                        // Process remaining
                        for i in 1..=seq.len() - k {
                            let prev_base = seq[i - 1];
                            let next_base = seq[i + k - 1];
                            rolling.slide(next_base, prev_base);

                            let packed =
                                std::cmp::min(rolling.forward(), rolling.reverse_complement());

                            *local_counts.entry(packed).or_insert(0) += 1;
                            let idx = self.map_kmer_to_neuron(packed);
                            currents[idx] += 1;
                            local_unique[idx] = true;
                        }
                    } else {
                        // Original method
                        for window in seq.windows(k) {
                            let packed = crate::utils::pack_kmer(window);
                            *local_counts.entry(packed).or_insert(0) += 1;

                            let idx = self.map_kmer_to_neuron(packed);
                            currents[idx] += 1;
                            local_unique[idx] = true;
                        }
                    }

                    maps.push(local_counts);
                    (currents, maps)
                },
            )
            .reduce(
                || (vec![0u64; pool_size], Vec::new()),
                |(mut currents1, mut maps1), (currents2, maps2)| {
                    for (i, &val) in currents2.iter().enumerate() {
                        currents1[i] += val;
                    }
                    maps1.extend(maps2);
                    (currents1, maps1)
                },
            );

        // Rest is the same as before
        self.counts.clear();
        for local_map in local_count_maps {
            for (kmer, cnt) in local_map {
                self.counts
                    .entry(kmer)
                    .or_insert(AtomicU32::new(0))
                    .fetch_add(cnt, Ordering::Relaxed);
            }
        }

        self.kmer_per_neuron.clear();
        for entry in self.counts.iter() {
            let kmer = *entry.key();
            let idx = self.map_kmer_to_neuron(kmer);
            *self.kmer_per_neuron.entry(idx).or_insert(0) += 1;
        }

        for (idx, &current) in total_currents.iter().enumerate() {
            self.neuron_currents[idx].store(current, Ordering::Relaxed);
        }

        // DEBUG: Check total current in in-memory version
        let total_current_sum: u64 = self
            .neuron_currents
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .sum();
        println!("  In-memory total current: {}", total_current_sum);

        // Spike simulation
        for (idx, neuron) in self.neurons.iter_mut().enumerate() {
            let total_current = total_currents[idx] as f64;
            if total_current == 0.0 {
                continue;
            }

            let per_step = total_current / self.steps as f64;

            for _ in 0..self.steps {
                if neuron.update(per_step as f32) {
                    self.energy.add_spike(self.spike_cost);
                }
            }
        }
    }

    pub fn process_sequence(&mut self, seq: &[u8]) {
        let mut local_unique = vec![false; self.pool_size];

        if seq.len() < self.k {
            return; // Skip sequences shorter than k
        }

        if self.use_canonical {
            // Use rolling hash + canonical
            let mut rolling = RollingKmerHash::new(self.k);

            // Process first k-mer
            rolling.init(&seq[0..self.k]);
            let canonical = rolling.canonical();

            let idx = self.map_kmer_to_neuron(canonical);
            self.counts
                .entry(canonical)
                .or_insert(AtomicU32::new(0))
                .fetch_add(1, Ordering::Relaxed);
            self.neuron_currents[idx].fetch_add(1, Ordering::Relaxed);
            local_unique[idx] = true;

            // Process remaining k-mers
            for i in 1..=seq.len() - self.k {
                let prev_base = seq[i - 1];
                let next_base = seq[i + self.k - 1];
                rolling.slide(next_base, prev_base);

                let canonical = rolling.canonical();

                let idx = self.map_kmer_to_neuron(canonical);
                self.counts
                    .entry(canonical)
                    .or_insert(AtomicU32::new(0))
                    .fetch_add(1, Ordering::Relaxed);
                self.neuron_currents[idx].fetch_add(1, Ordering::Relaxed);
                local_unique[idx] = true;
            }
        } else {
            // Original method (no canonical)
            for window in seq.windows(self.k) {
                let packed = crate::utils::pack_kmer(window);
                let idx = self.map_kmer_to_neuron(packed);

                self.counts
                    .entry(packed)
                    .or_insert(AtomicU32::new(0))
                    .fetch_add(1, Ordering::Relaxed);

                self.neuron_currents[idx].fetch_add(1, Ordering::Relaxed);
                local_unique[idx] = true;
            }
        }

        // Update unique tracking (same as before)
        for (idx, unique) in local_unique.iter().enumerate() {
            if *unique {
                *self.kmer_per_neuron.entry(idx).or_insert(0) += 1;
            }
        }

        // Spike simulation (same as before)
        for (idx, neuron) in self.neurons.iter_mut().enumerate() {
            let current = self.neuron_currents[idx].load(Ordering::Relaxed) as f64;
            if current > 0.0 && neuron.update(current as f32) {
                self.energy.add_spike(self.spike_cost);
            }
            self.neuron_currents[idx].store(0, Ordering::Relaxed);
        }
    }

    /// Process a FASTA/FASTQ file in streaming fashion with parallel workers
    /// Memory usage: O(num_threads * max_seq_len) instead of O(total_bases)
    pub fn process_file_streaming(&mut self, path: &str) -> NeuroResult<()> {
        println!("Starting streaming for: {}", path);
        let k = self.k;
        let pool_size = self.pool_size;
        let use_canonical = self.use_canonical;

        // Create channels using crossbeam
        let (seq_tx, seq_rx) = unbounded::<Vec<u8>>();
        let (result_tx, result_rx) = unbounded::<(
            Vec<u64>,          // neuron currents
            HashMap<u64, u32>, // local counts
            Vec<(usize, u32)>, // unique k-mer tracking (neuron_idx, count)
        )>();

        // Number of worker threads
        let num_workers = rayon::current_num_threads();
        println!("  Starting {} worker threads", num_workers);

        // Start worker threads
        let mut workers = Vec::with_capacity(num_workers);
        for worker_id in 0..num_workers {
            let seq_rx = seq_rx.clone();
            let result_tx = result_tx.clone();
            let k = k;
            let pool_size = pool_size;
            let use_canonical = use_canonical;

            workers.push(thread::spawn(move || {
                // Create a local hasher for this thread
                println!("    Worker {} started", worker_id);

                // OPTIMIZATION 1: Pre-allocate vectors with capacity
                let mut local_currents = vec![0u64; pool_size];
                let mut local_counts = HashMap::with_capacity(10000); // Pre-allocate for 10k k-mers
                let mut local_unique = vec![false; pool_size];

                // OPTIMIZATION 2: Pre-allocate the unique_updates vector to avoid repeated allocations
                let mut unique_updates = Vec::with_capacity(pool_size);

                let mut seq_count = 0;
                let mut total_seqs = 0;

                let map_kmer_to_neuron = |packed: u64| -> usize {
                    let mut hasher = SipHasher13::new_with_keys(0, 0);
                    packed.hash(&mut hasher);
                    (hasher.finish() % pool_size as u64) as usize
                };

                while let Ok(seq) = seq_rx.recv() {
                    total_seqs += 1;
                    seq_count += 1;

                    if seq.len() < k {
                        continue;
                    }

                    // Reset unique tracking for this sequence
                    for u in &mut local_unique {
                        *u = false;
                    }

                    if use_canonical {
                        // OPTIMIZATION 3: Reuse RollingKmerHash by creating it once outside the loop
                        // But we can't because each sequence needs fresh state
                        // Instead, we'll keep as is
                        let mut rolling = RollingKmerHash::new(k);

                        // Process first k-mer
                        rolling.init(&seq[0..k]);
                        let canonical = rolling.canonical();

                        *local_counts.entry(canonical).or_insert(0) += 1;
                        let idx = map_kmer_to_neuron(canonical);
                        local_currents[idx] += 1;
                        local_unique[idx] = true;

                        // Process remaining k-mers
                        for i in 1..=seq.len() - k {
                            let prev_base = seq[i - 1];
                            let next_base = seq[i + k - 1];
                            rolling.slide(next_base, prev_base);

                            let canonical = rolling.canonical();

                            *local_counts.entry(canonical).or_insert(0) += 1;
                            let idx = map_kmer_to_neuron(canonical);
                            local_currents[idx] += 1;
                            local_unique[idx] = true;
                        }
                    } else {
                        for window in seq.windows(k) {
                            let packed = crate::utils::pack_kmer(window);

                            *local_counts.entry(packed).or_insert(0) += 1;
                            let idx = map_kmer_to_neuron(packed);
                            local_currents[idx] += 1;
                            local_unique[idx] = true;
                        }
                    }

                    // Periodically send results
                    if seq_count % 1000 == 0 {
                        // OPTIMIZATION: Increased from 100 to 1000
                        // OPTIMIZATION: Clear and reuse unique_updates instead of creating new Vec
                        unique_updates.clear();
                        for (idx, unique) in local_unique.iter().enumerate() {
                            if *unique {
                                unique_updates.push((idx, 1));
                            }
                        }

                        if result_tx
                            .send((
                                local_currents.clone(),
                                local_counts.clone(),
                                unique_updates.clone(),
                            ))
                            .is_err()
                        {
                            break;
                        }

                        // Reset accumulators (but keep allocated memory)
                        for c in &mut local_currents {
                            *c = 0;
                        }
                        local_counts.clear(); // Keeps capacity
                    }
                }

                println!(
                    "    Worker {} finished after {} sequences",
                    worker_id, total_seqs
                );

                // Send final results
                if !local_currents.iter().all(|&c| c == 0) || !local_counts.is_empty() {
                    unique_updates.clear();
                    for (idx, unique) in local_unique.iter().enumerate() {
                        if *unique {
                            unique_updates.push((idx, 1));
                        }
                    }

                    let _ = result_tx.send((local_currents, local_counts, unique_updates));
                }
            }));
        }

        // DROP THE ORIGINAL RESULT_TX HERE - THIS IS CRITICAL!
        drop(result_tx);

        // Producer thread
        let path = path.to_string();
        let producer = thread::spawn(move || {
            println!("  Producer started");
            let mut seq_count = 0;
            let reader = crate::utils::stream_sequences(&path)?;
            for seq in reader {
                seq_count += 1;
                if seq_tx.send(seq).is_err() {
                    println!("  Producer: channel closed, stopping");
                    break;
                }
            }
            println!("  Producer finished after {} sequences", seq_count);
            drop(seq_tx);
            NeuroResult::Ok(())
        });

        // Aggregator
        println!("  Aggregator started");
        self.counts.clear();
        self.kmer_per_neuron.clear();
        for neuron_current in &self.neuron_currents {
            neuron_current.store(0, Ordering::Relaxed);
        }

        let mut total_currents = vec![0u64; self.pool_size];
        let mut result_count = 0;

        while let Ok((currents, counts, unique_updates)) = result_rx.recv() {
            result_count += 1;

            // Accumulate currents
            for (i, &val) in currents.iter().enumerate() {
                total_currents[i] += val;
            }

            // Merge counts
            for (kmer, cnt) in counts {
                self.counts
                    .entry(kmer)
                    .or_insert(AtomicU32::new(0))
                    .fetch_add(cnt, Ordering::Relaxed);
            }

            // Update unique k-mer tracking
            for (idx, _) in unique_updates {
                *self.kmer_per_neuron.entry(idx).or_insert(0) += 1;
            }
        }
        println!(
            "  Aggregator finished after {} result batches",
            result_count
        );
        // Wait for producer to finish
        producer.join().unwrap()?;

        // Wait for all workers
        for (i, worker) in workers.into_iter().enumerate() {
            println!("  Waiting for worker {} to join...", i);
            let _ = worker.join();
        }

        // Store final currents
        for (idx, &current) in total_currents.iter().enumerate() {
            self.neuron_currents[idx].store(current, Ordering::Relaxed);
        }

        // DEBUG: Check total current
        let total_current_sum: u64 = self
            .neuron_currents
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .sum();
        println!(
            "  Total accumulated current before SIMD: {}",
            total_current_sum
        );

        // Run spike simulation
        println!("  Running SIMD spike simulation...");
        self.simulate_spikes_simd();
        // println!("  Running parallel spike simulation...");
        // self.simulate_spikes_parallel();
        println!("  Streaming complete!");
        Ok(())
    }
    /// Run spike simulation on accumulated currents
    fn simulate_spikes(&mut self) {
        for (idx, neuron) in self.neurons.iter_mut().enumerate() {
            let total_current = self.neuron_currents[idx].load(Ordering::Relaxed) as f64;
            if total_current == 0.0 {
                continue;
            }

            // Reduce steps from 1000 to 500 for 2x speedup
            // Spike count will be halved but relative comparisons remain valid
            let per_step = total_current / self.steps as f64;

            for _ in 0..self.steps {
                if neuron.update(per_step as f32) {
                    self.energy.add_spike(self.spike_cost);
                }
            }
        }
    }

    fn simulate_spikes_parallel(&mut self) {
        use rayon::prelude::*;

        // Process neurons in parallel batches
        let batch_size = 1000;
        self.neurons
            .par_chunks_mut(batch_size)
            .enumerate()
            .for_each(|(chunk_idx, neuron_chunk)| {
                let start_idx = chunk_idx * batch_size;
                for (offset, neuron) in neuron_chunk.iter_mut().enumerate() {
                    let idx = start_idx + offset;
                    let total_current = self.neuron_currents[idx].load(Ordering::Relaxed) as f64;
                    if total_current == 0.0 {
                        continue;
                    }

                    let per_step = total_current / self.steps as f64;
                    let mut local_spikes = 0u64;

                    for _ in 0..self.steps {
                        if neuron.update(per_step as f32) {
                            local_spikes += 1;
                        }
                    }
                    self.energy
                        .total_spikes
                        .fetch_add(local_spikes, Ordering::Relaxed);
                    let cost_fixed = local_spikes * (self.spike_cost * 1000.0) as u64;
                    self.energy
                        .total_energy
                        .fetch_add(cost_fixed, Ordering::Relaxed);
                }
            });
    }

    /// SIMD-accelerated spike simulation - MATCHES SCALAR PRECISION
    fn simulate_spikes_simd(&mut self) {
        use std::sync::atomic::Ordering;

        let steps = self.steps as u64;
        if steps == 0 {
            return;
        }

        let spike_cost = self.spike_cost;

        // Pre-load persistent batch arrays with current neuron state
        for i in 0..self.pool_size {
            let neuron = &self.neurons[i];
            self.batch_voltages[i] = neuron.voltage;
            self.batch_thresholds[i] = neuron.threshold;
            self.batch_leaks[i] = neuron.leak;
            self.batch_refractory[i] = neuron.refractory_ticks;
        }

        // Store total currents for each neuron
        let mut total_currents_per_neuron = vec![0u64; self.pool_size];
        for i in 0..self.pool_size {
            total_currents_per_neuron[i] = self.neuron_currents[i].load(Ordering::Relaxed);
        }

        let mut per_neuron_spikes = vec![0u64; self.pool_size];
        let mut total_spikes = 0u64;
        let mut total_energy_fixed = 0u64;

        // Process in batches of 8 neurons
        for batch_start in (0..self.pool_size).step_by(8) {
            let batch_end = (batch_start + 8).min(self.pool_size);
            let batch_size = batch_end - batch_start;

            // Fixed-size arrays
            let mut v_batch = [0.0f32; 8];
            let mut t_batch = [0.0f32; 8];
            let mut l_batch = [0.0f32; 8];
            let mut r_batch = [0u32; 8];
            let mut current_values = [0u64; 8]; // Store raw currents for this batch

            // Copy initial state for this batch
            for j in 0..batch_size {
                let idx = batch_start + j;
                v_batch[j] = self.batch_voltages[idx];
                t_batch[j] = self.batch_thresholds[idx];
                l_batch[j] = self.batch_leaks[idx];
                r_batch[j] = self.batch_refractory[idx];
                current_values[j] = total_currents_per_neuron[idx];
            }

            let mut batch_spikes = [0u64; 8];

            // Simulate all timesteps
            for _ in 0..steps {
                // Calculate per-step current INSIDE the loop (like scalar version)
                let mut c_batch = [0.0f32; 8];
                for j in 0..batch_size {
                    if current_values[j] > 0 {
                        // Use f64 division for exact match with scalar
                        let per_step_f64 = current_values[j] as f64 / steps as f64;
                        c_batch[j] = per_step_f64 as f32;
                    }
                }

                let spikes_this_step = LifNeuron::update_batch_simd(
                    &mut v_batch,
                    &t_batch,
                    &l_batch,
                    &c_batch,
                    &mut r_batch,
                    self.refractory,
                );

                for j in 0..batch_size {
                    batch_spikes[j] += spikes_this_step[j] as u64;
                }
            }

            // DEBUG for neuron 105413
            if batch_start <= 105413 && 105413 < batch_end {
                let local_idx = 105413 - batch_start;
                println!(
                    "    DEBUG neuron 105413: spikes = {}, final voltage = {:.4}, refractory = {}, current = {}",
                    batch_spikes[local_idx],
                    v_batch[local_idx],
                    r_batch[local_idx],
                    current_values[local_idx]
                );
            }

            // Accumulate results
            for j in 0..batch_size {
                let idx = batch_start + j;
                self.batch_voltages[idx] = v_batch[j];
                self.batch_refractory[idx] = r_batch[j];

                let spikes = batch_spikes[j];
                per_neuron_spikes[idx] += spikes;
                total_spikes += spikes;
                total_energy_fixed += spikes * (spike_cost * 1000.0) as u64;
            }
        }

        // Final write-back to neurons
        for i in 0..self.pool_size {
            let neuron = &mut self.neurons[i];
            neuron.voltage = self.batch_voltages[i];
            neuron.refractory_ticks = self.batch_refractory[i];
            neuron.spike_count += per_neuron_spikes[i];
        }

        self.energy
            .total_spikes
            .fetch_add(total_spikes, Ordering::Relaxed);
        self.energy
            .total_energy
            .fetch_add(total_energy_fixed, Ordering::Relaxed);

        println!("    SIMD total spikes: {}", total_spikes);
    }

    /// Get top spiking neurons (proxy for most abundant k-mer groups)
    pub fn top_abundant_neurons(&self, top_n: usize) -> Vec<(usize, u64, u32)> {
        let mut stats: Vec<_> = self
            .neurons
            .iter()
            .enumerate()
            .map(|(idx, neuron)| {
                let uniques = self.kmer_per_neuron.get(&idx).map(|r| *r).unwrap_or(0);
                (idx, neuron.spike_count, uniques)
            })
            .collect();
        stats.sort_by(|a, b| b.1.cmp(&a.1));
        stats.into_iter().take(top_n).collect()
    }

    pub fn get_count(&self, kmer: u64) -> Option<u32> {
        self.counts
            .get(&kmer)
            .map(|entry| entry.value().load(Ordering::Relaxed))

        // let idx = self.map_kmer_to_neuron(kmer);
        // self.kmer_per_neuron.get(&idx).map(|v| *v)
    }

    pub fn energy_used(&self) -> f64 {
        self.energy.total_energy()
    }

    pub fn set_steps(&mut self, steps: usize) {
        self.steps = steps;
    }

    /// Get the current number of simulation steps
    pub fn get_steps(&self) -> usize {
        self.steps
    }

    pub fn simulate_spikes_auto(&mut self) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                println!("  Using AVX2 SIMD (8-wide)");
                self.simulate_spikes_simd();
            } else {
                println!("  Using parallel scalar (AVX2 not available)");
                self.simulate_spikes_parallel();
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            println!("  Using parallel scalar");
            self.simulate_spikes_parallel();
        }
    }
}
