// src/spiking_hash.rs
// Uses borrowed LIF for event-driven counting
// One neuron per unique k-mer (or hash-mapped for efficiency)

use crate::models::{EnergyTracker, LifNeuron};
use dashmap::DashMap;
use rayon::prelude::*;
use siphasher::sip::SipHasher13; // Add to Cargo.toml: siphasher = "0.3"
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct SpikingKmerCounter {
    neurons: Vec<LifNeuron>, // Fixed pool (e.g., 1M neurons)
    pub energy: EnergyTracker,
    pub k: usize,
    pool_size: usize, // NEW: Fixed neuron count
    spike_cost: f64,
    threshold: f64,
    leak: f64,
    refractory: u32,
    // pub counts: DashMap<u64, u32>,   // Thread-safe hash map
    spike_to_kmer: Vec<Vec<u64>>,    // Track which k-mers mapped to each neuron
    neuron_currents: Vec<AtomicU64>, // Thread-safe atomic for parallel accumulation
    // Optional: Keep track of unique k-mers per neuron for collision stats
    pub kmer_per_neuron: DashMap<usize, u32>,
}

impl SpikingKmerCounter {
    pub fn new(
        k: usize,
        threshold: f64,
        leak: f64,
        refractory: u32,
        spike_cost: f64,
        pool_size: usize,
    ) -> Self {
        let mut neurons = Vec::with_capacity(pool_size);
        let mut spike_to_kmer = Vec::with_capacity(pool_size);
        let mut neuron_currents = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            neurons.push(LifNeuron::new(threshold, leak, refractory));
            neuron_currents.push(AtomicU64::new(0));
        }
        Self {
            neurons,
            energy: EnergyTracker::default(),
            k,
            pool_size,
            spike_cost,
            neuron_currents: (0..pool_size).map(|_| AtomicU64::new(0)).collect(),
            kmer_per_neuron: DashMap::new(),
            spike_to_kmer,
            threshold,
            leak,
            refractory,
        }
    }

    /* pub fn process_sequence_parallel(&mut self, seqs: &[Vec<u8>]) {
        // First pass: count k-mer occurrences per neuron
        let neuron_currents = seqs
            .par_iter()
            .map(|seq| {
                let mut local_currents = vec![0.0; self.pool_size];
                for window in seq.windows(self.k) {
                    let packed = crate::utils::pack_kmer(window);
                    let neuron_idx = self.map_kmer_to_neuron(packed);
                    local_currents[neuron_idx] += 1.0; // Each occurrence adds current
                }
                local_currents
            })
            .reduce(
                || vec![0.0; self.pool_size],
                |mut acc, local| {
                    for (i, &val) in local.iter().enumerate() {
                        acc[i] += val;
                    }
                    acc
                },
            );

        // Second pass: run neurons with accumulated current
        for (i, current) in neuron_currents.iter().enumerate() {
            let neuron = &mut self.neurons[i];

            // Distribute current over time steps (simplified as one big step)
            let total_steps = 1000; // Simulate 1000 time steps
            let per_step = current / total_steps as f64;

            for _ in 0..total_steps {
                if neuron.update(per_step) {
                    self.energy.total_spikes += 1;
                    self.energy.total_energy += self.spike_cost;

                    // Spike means "activate all k-mers in this neuron"
                    for &kmer in &self.spike_to_kmer[i] {
                        *self.counts.entry(kmer).or_insert(0) += 1;
                    }
                }
            }
        }
    }
 */
    fn map_kmer_to_neuron(&self, packed: u64) -> usize {
        let mut hasher = SipHasher13::new_with_keys(0, 0);
        packed.hash(&mut hasher);
        (hasher.finish() % self.pool_size as u64) as usize
    }

    pub fn process_parallel(&mut self, seqs: &[Vec<u8>]) {
        // Parallel accumulation of currents + unique k-mer tracking
        seqs.par_iter().for_each(|seq| {
            let mut local_unique = vec![false; self.pool_size];
            for window in seq.windows(self.k) {
                let packed = crate::utils::pack_kmer(window);
                let idx = self.map_kmer_to_neuron(packed);

                // Accumulate current atomically
                self.neuron_currents[idx].fetch_add(1, Ordering::Relaxed);

                // Track unique (local to avoid lock contention)
                local_unique[idx] = true;
            }
            // Update global unique count (thread-safe)
            for (idx, unique) in local_unique.iter().enumerate() {
                if *unique {
                    *self.kmer_per_neuron
                        .entry(idx)
                        .or_insert(0) += 1;
                }
            }
        });

        // Serial simulation: distribute accumulated current over time steps
        for (idx, neuron) in self.neurons.iter_mut().enumerate() {
            let total_current = self.neuron_currents[idx].load(Ordering::Relaxed) as f64;
            if total_current == 0.0 {
                continue;
            }

            let steps = 1000usize; // Simulate over fixed time steps
            let per_step = total_current / steps as f64;

            for _ in 0..steps {
                if neuron.update(per_step) {
                    self.energy.total_spikes += 1;
                    self.energy.total_energy += self.spike_cost;
                }
            }
        }
    }

    pub fn process_sequence(&mut self, seq: &[u8]) {
        let mut local_unique = vec![false; self.pool_size];
        for window in seq.windows(self.k) {
            let packed = crate::utils::pack_kmer(window);
            let idx = self.map_kmer_to_neuron(packed);

            self.neuron_currents[idx].fetch_add(1, Ordering::Relaxed);
            local_unique[idx] = true;
        }
        for (idx, unique) in local_unique.iter().enumerate() {
            if *unique {
                *self.kmer_per_neuron.entry(idx).or_insert(0) += 1;
            }
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
        let idx = self.map_kmer_to_neuron(kmer);
        self.kmer_per_neuron.get(&idx).map(|v| *v)
    }

    pub fn energy_used(&self) -> f64 {
        self.energy.total_energy
    }
}
