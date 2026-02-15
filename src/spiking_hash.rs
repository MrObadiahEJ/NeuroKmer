// src/spiking_hash.rs
// Uses borrowed LIF for event-driven counting
// One neuron per unique k-mer (or hash-mapped for efficiency)

use crate::models::{EnergyTracker, LifNeuron};
use dashmap::DashMap;
use rayon::prelude::*;
use siphasher::sip::SipHasher13; // Add to Cargo.toml: siphasher = "0.3"
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

pub struct SpikingKmerCounter {
    neurons: Vec<LifNeuron>, // Fixed pool (e.g., 1M neurons)
    pub energy: EnergyTracker,
    pub k: usize,
    threshold: f64,
    leak: f64,
    refractory: u32, //
    spike_cost: f64,
    pool_size: usize,              // NEW: Fixed neuron count
    pub counts: DashMap<u64, u32>, // Thread-safe hash map
    spike_to_kmer: Vec<Vec<u64>>,  // Track which k-mers mapped to each neuron
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
        for _ in 0..pool_size {
            neurons.push(LifNeuron::new(threshold, leak, refractory));
            spike_to_kmer.push(Vec::new());
        }
        Self {
            neurons,
            energy: EnergyTracker::default(),
            k,
            threshold,
            leak,
            refractory,
            spike_cost,
            pool_size,
            counts: DashMap::new(),
            spike_to_kmer,
        }
    }

    pub fn process_sequence_parallel(&mut self, seqs: &[Vec<u8>]) {
        // First pass: count k-mer occurrences per neuron
        let neuron_currents = seqs.par_iter()
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

    fn map_kmer_to_neuron(&self, packed: u64) -> usize {
        let mut hasher = SipHasher13::new_with_keys(0, 0);
        packed.hash(&mut hasher);
        (hasher.finish() % self.pool_size as u64) as usize
    }

    pub fn process_sequence(&mut self, seq: &[u8]) {
        for window in seq.windows(self.k) {
            let packed = crate::utils::pack_kmer(window);
            let neuron_idx = self.map_kmer_to_neuron(packed);

            // Record that this k-mer maps to this neuron
            if !self.spike_to_kmer[neuron_idx].contains(&packed) {
                self.spike_to_kmer[neuron_idx].push(packed);
            }

            let neuron = &mut self.neurons[neuron_idx];

            if neuron.update(1.0) {
                self.energy.total_spikes += 1;
                self.energy.total_energy += self.spike_cost;

                // When neuron spikes, increment ALL k-mers that map to it
                // This is a simplification – you'll refine this in Step 2
                for &kmer in &self.spike_to_kmer[neuron_idx] {
                    *self.counts.entry(kmer).or_insert(0) += 1;
                }
            }
        }
    }

    pub fn get_count(&self, kmer: u64) -> Option<u32> {
        self.counts.get(&kmer).map(|v| *v)
    }

    pub fn energy_used(&self) -> f64 {
        self.energy.total_energy
    }
}
