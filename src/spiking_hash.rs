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
        let batch_send_interval = 10000;

        // Create channels using crossbeam
        let (seq_tx, seq_rx) = unbounded::<Vec<u8>>();
        let (result_tx, result_rx) = unbounded::<(
            Vec<u64>,        // neuron indices (compact)
            Vec<u64>,        // neuron currents (compact)
            Vec<(u64, u32)>, // k-mer counts (kmer_hash, count) - only non-zero, aggregated
        )>();

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
                println!("    Worker {} started", worker_id);

                // FAST PATH: Vec<u64> for neuron currents (no HashMap overhead)
                let mut local_currents = vec![0u64; pool_size];
                // SLOW PATH: Only for k-mers that actually appear (sparse)
                let mut local_kmer_counts: HashMap<u64, u32> = HashMap::with_capacity(1024);
                let mut seq_count = 0;

                let map_kmer_to_neuron = |packed: u64| -> usize {
                    let mut hasher = SipHasher13::new_with_keys(0, 0);
                    packed.hash(&mut hasher);
                    (hasher.finish() % pool_size as u64) as usize
                };

                while let Ok(seq) = seq_rx.recv() {
                    seq_count += 1;

                    if seq.len() < k {
                        continue;
                    }

                    if use_canonical {
                        let mut rolling = RollingKmerHash::new(k);
                        rolling.init(&seq[0..k]);
                        let canonical = rolling.canonical();

                        let idx = map_kmer_to_neuron(canonical);
                        local_currents[idx] += 1;
                        *local_kmer_counts.entry(canonical).or_insert(0) += 1;

                        for i in 1..=seq.len() - k {
                            let prev_base = seq[i - 1];
                            let next_base = seq[i + k - 1];
                            rolling.slide(next_base, prev_base);
                            let canonical = rolling.canonical();

                            let idx = map_kmer_to_neuron(canonical);
                            local_currents[idx] += 1;
                            *local_kmer_counts.entry(canonical).or_insert(0) += 1;
                        }
                    } else {
                        for window in seq.windows(k) {
                            let packed = crate::utils::pack_kmer(window);
                            let idx = map_kmer_to_neuron(packed);
                            local_currents[idx] += 1;
                            *local_kmer_counts.entry(packed).or_insert(0) += 1;
                        }
                    }

                    // Send results periodically
                    if seq_count % batch_send_interval == 0 {
                        // Compact neuron currents (FAST)
                        let mut compact_currents = Vec::with_capacity(pool_size / 10);
                        let mut indices = Vec::with_capacity(pool_size / 10);

                        for (idx, &val) in local_currents.iter().enumerate() {
                            if val > 0 {
                                compact_currents.push(val);
                                indices.push(idx as u64);
                            }
                        }

                        // Convert HashMap to Vec for sending (avoid serialization overhead)
                        let kmer_vec: Vec<(u64, u32)> = local_kmer_counts.drain().collect();

                        if result_tx
                            .send((indices, compact_currents, kmer_vec))
                            .is_err()
                        {
                            break;
                        }

                        local_currents.fill(0);
                    }
                }

                println!(
                    "    Worker {} finished after {} sequences",
                    worker_id, seq_count
                );

                // Send final results
                let mut compact_currents = Vec::with_capacity(pool_size / 10);
                let mut indices = Vec::with_capacity(pool_size / 10);

                for (idx, &val) in local_currents.iter().enumerate() {
                    if val > 0 {
                        compact_currents.push(val);
                        indices.push(idx as u64);
                    }
                }

                let kmer_vec: Vec<(u64, u32)> = local_kmer_counts.drain().collect();

                if !compact_currents.is_empty() || !kmer_vec.is_empty() {
                    let _ = result_tx.send((indices, compact_currents, kmer_vec));
                }
            }));
        }

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
                    break;
                }
            }
            println!("  Producer finished after {} sequences", seq_count);
            drop(seq_tx);
            NeuroResult::Ok(())
        });

        // Aggregator - handles both fast neuron currents AND k-mer counts
        println!("  Aggregator started");
        self.counts.clear();
        self.kmer_per_neuron.clear();

        let mut total_currents = vec![0u64; self.pool_size];
        let mut result_count = 0;

        while let Ok((indices, values, kmer_counts)) = result_rx.recv() {
            result_count += 1;

            // Fast accumulation for neuron currents
            for (i, &idx_u64) in indices.iter().enumerate() {
                let idx = idx_u64 as usize;
                total_currents[idx] += values[i];
            }

            // Merge k-mer counts (slower but necessary for get_count)
            for (kmer, count) in kmer_counts {
                self.counts
                    .entry(kmer)
                    .or_insert(AtomicU32::new(0))
                    .fetch_add(count, Ordering::Relaxed);
            }
        }

        println!(
            "  Aggregator finished after {} result batches",
            result_count
        );

        producer.join().unwrap()?;

        for (i, worker) in workers.into_iter().enumerate() {
            println!("  Waiting for worker {} to join...", i);
            let _ = worker.join();
        }

        // Store final currents
        for (idx, &current) in total_currents.iter().enumerate() {
            self.neuron_currents[idx].store(current, Ordering::Relaxed);
        }

        // Rebuild kmer_per_neuron from counts
        self.kmer_per_neuron.clear();
        for entry in self.counts.iter() {
            let kmer = *entry.key();
            let idx = self.map_kmer_to_neuron(kmer);
            *self.kmer_per_neuron.entry(idx).or_insert(0) += 1;
        }

        let total_current_sum: u64 = total_currents.iter().sum();
        println!(
            "  Total accumulated current before SIMD: {}",
            total_current_sum
        );

        println!("  Running SIMD spike simulation...");
        self.simulate_spikes_simd();
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

    /// SIMD-accelerated spike simulation - FULLY VECTORIZED
    fn simulate_spikes_simd(&mut self) {
        use std::arch::x86_64::*;
        use std::sync::atomic::Ordering;

        let steps = self.steps;
        if steps == 0 {
            return;
        }

        let spike_cost = self.spike_cost;
        let steps_f64 = steps as f64;
        let refractory_period = self.refractory;

        unsafe {
            let mut total_spikes: u64 = 0;

            // Process in batches of 8 neurons
            for batch_start in (0..self.pool_size).step_by(8) {
                let batch_end = (batch_start + 8).min(self.pool_size);
                let batch_size = batch_end - batch_start;

                if batch_size == 0 {
                    continue;
                }

                // Load neuron state
                let mut v = _mm256_setzero_ps();
                let t = _mm256_set1_ps(self.threshold);
                let l = _mm256_set1_ps(self.leak);
                let mut r = [0u32; 8];
                let mut spike_counts = [0u64; 8];

                // Initialize from neurons
                let mut v_arr = [0.0f32; 8];
                for j in 0..batch_size {
                    let idx = batch_start + j;
                    v_arr[j] = self.neurons[idx].voltage;
                    r[j] = self.neurons[idx].refractory_ticks;
                }
                v = _mm256_loadu_ps(v_arr.as_ptr());

                // Get currents for this batch
                let mut currents = [0.0f32; 8];
                for j in 0..batch_size {
                    let idx = batch_start + j;
                    let total_c = self.neuron_currents[idx].load(Ordering::Relaxed) as f64;
                    currents[j] = (total_c / steps_f64) as f32;
                }
                let c_vec = _mm256_loadu_ps(currents.as_ptr());

                // SIMD simulation loop - no function calls, all inline
                for _ in 0..steps {
                    // Build active mask: refractory == 0
                    let mut active_arr = [0u32; 8];
                    for j in 0..8 {
                        if r[j] == 0 {
                            active_arr[j] = 0xFFFFFFFFu32;
                        }
                    }
                    let active_mask = _mm256_loadu_ps(active_arr.as_ptr() as *const f32);

                    // Update: v = active ? (v * leak + current) : v
                    let v_leaked = _mm256_mul_ps(v, l);
                    let v_new = _mm256_add_ps(v_leaked, c_vec);
                    v = _mm256_blendv_ps(v, v_new, active_mask);

                    // Spike detection
                    let max_val = _mm256_set1_ps(f32::MAX);
                    let t_eff = _mm256_blendv_ps(max_val, t, active_mask);
                    let spike_mask_vec = _mm256_cmp_ps(v, t_eff, _CMP_GE_OQ);
                    let spike_mask = _mm256_movemask_ps(spike_mask_vec) as u32;

                    // Reset voltages where spike occurred
                    let zero = _mm256_setzero_ps();
                    v = _mm256_blendv_ps(v, zero, spike_mask_vec);

                    // Update refractory and count spikes (this is the unavoidable scalar part)
                    // But we can make it branchless and fast
                    for j in 0..batch_size {
                        let is_refractory = (r[j] > 0) as u32;
                        let has_spike = (spike_mask >> j) & 1;

                        r[j] = if is_refractory == 1 {
                            r[j] - 1
                        } else if has_spike == 1 {
                            refractory_period
                        } else {
                            0
                        };

                        spike_counts[j] += has_spike as u64;
                    }
                }

                // Write back results
                _mm256_storeu_ps(v_arr.as_mut_ptr(), v);
                for j in 0..batch_size {
                    let idx = batch_start + j;
                    self.neurons[idx].voltage = v_arr[j];
                    self.neurons[idx].refractory_ticks = r[j];
                    self.neurons[idx].spike_count += spike_counts[j];
                    total_spikes += spike_counts[j];
                }
            }

            let total_energy_fixed = total_spikes * (spike_cost * 1000.0) as u64;
            self.energy
                .total_spikes
                .fetch_add(total_spikes, Ordering::Relaxed);
            self.energy
                .total_energy
                .fetch_add(total_energy_fixed, Ordering::Relaxed);

            println!("    SIMD total spikes: {}", total_spikes);
        }
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
