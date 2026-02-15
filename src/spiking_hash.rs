// src/spiking_hash.rs
// Uses borrowed LIF for event-driven counting
// One neuron per unique k-mer (or hash-mapped for efficiency)

use std::collections::HashMap;
use crate::models::{LifNeuron, EnergyTracker};
use std::hash::{Hash, Hasher};
use siphasher::sip::SipHasher13;  // Add to Cargo.toml: siphasher = "0.3"

pub struct SpikingKmerCounter {
    neurons: Vec<LifNeuron>,       // Fixed pool (e.g., 1M neurons)
    pub energy: EnergyTracker,
    pub k: usize,
    threshold: f64,
    leak: f64,
    refractory: u32,               // 
    spike_cost: f64,
    pool_size: usize,              // NEW: Fixed neuron count
}

impl SpikingKmerCounter {
    pub fn new(k: usize, threshold: f64, leak: f64, refractory: u32, spike_cost: f64, pool_size: usize) -> Self {
        let mut neurons = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            neurons.push(LifNeuron::new(threshold, leak, refractory));
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
            let neuron = &mut self.neurons[neuron_idx];

            if neuron.update(1.0) {
                self.energy.total_spikes += 1;
                self.energy.total_energy += self.spike_cost;
            }
        }
    }

    pub fn energy_used(&self) -> f64 {
        self.energy.total_energy
    }
}
