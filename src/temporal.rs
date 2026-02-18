// src/temporal.rs
// Temporal coding: spike timing encodes k-mer identity and abundance

use std::collections::HashMap;
use std::sync::atomic::Ordering;

/// Temporal code for a k-mer: when it spikes and how strong
#[derive(Debug, Clone, Copy)]
pub struct TemporalSpike {
    pub neuron_id: usize,
    pub time_step: u32, // When the spike occurred
    pub amplitude: f32, // Spike amplitude
}

/// Temporal coding scheme using time-to-first-spike (TTFS)
pub struct TemporalCoder {
    pub steps: usize,
    pub max_time: u32,
    pub spike_times: HashMap<u64, TemporalSpike>,
    pub temporal_resolution: u32,
}

impl TemporalCoder {
    pub fn new(steps: usize) -> Self {
        Self {
            steps,
            max_time: steps as u32,
            spike_times: HashMap::new(),
            temporal_resolution: 1,
        }
    }

    /// Encode k-mer abundance as spike timing (TTFS)
    pub fn encode_count(&mut self, kmer: u64, count: u32) -> u32 {
        if count == 0 {
            return 0;
        }

        let log_count = (count as f32).ln_1p();
        let max_log = (self.max_time as f32).ln_1p();
        let normalized = log_count / max_log;
        let time_step = (self.max_time as f32 * (1.0 - normalized)).max(1.0) as u32;

        let spike = TemporalSpike {
            neuron_id: 0,
            time_step,
            amplitude: (count as f32).min(100.0),
        };

        self.spike_times.insert(kmer, spike);
        time_step
    }

    /// Rank-order coding
    pub fn rank_order_encode(&mut self, kmer_counts: &[(u64, u32)]) -> Vec<(u64, u32)> {
        let mut sorted = kmer_counts.to_vec();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        let mut result = Vec::with_capacity(sorted.len());
        for (rank, (kmer, count)) in sorted.iter().enumerate() {
            let time_step = (rank + 1).min(self.max_time as usize) as u32;

            let spike = TemporalSpike {
                neuron_id: 0,
                time_step,
                amplitude: *count as f32,
            };

            self.spike_times.insert(*kmer, spike);
            result.push((*kmer, time_step));
        }

        result
    }

    /// Get spike time for a k-mer
    pub fn get_spike_time(&self, kmer: u64) -> Option<u32> {
        self.spike_times.get(&kmer).map(|s| s.time_step)
    }

    /// Decode approximate count from spike time
    pub fn decode_timing(&self, kmer: u64) -> Option<f32> {
        self.spike_times.get(&kmer).map(|spike| {
            if spike.time_step == 0 {
                return 0.0;
            }
            let normalized = 1.0 - (spike.time_step as f32 / self.max_time as f32);
            let log_count = normalized * (self.max_time as f32).ln_1p();
            log_count.exp() - 1.0
        })
    }
}

/// Temporal coding integrated with spiking counter
pub struct TemporalSpikingCounter {
    pub counter: crate::spiking_hash::SpikingKmerCounter,
    pub temporal: TemporalCoder,
    pub use_temporal: bool,
}

impl TemporalSpikingCounter {
    pub fn new(
        k: usize,
        threshold: f32,
        leak: f32,
        refractory: u32,
        spike_cost: f64,
        pool_size: usize,
        use_canonical: bool,
        steps: usize,
    ) -> Self {
        Self {
            counter: crate::spiking_hash::SpikingKmerCounter::new(
                k,
                threshold,
                leak,
                refractory,
                spike_cost,
                pool_size,
                use_canonical,
            ),
            temporal: TemporalCoder::new(steps),
            use_temporal: true,
        }
    }

    /// Process with temporal coding
    pub fn process_with_temporal(&mut self, seqs: &[Vec<u8>]) {
        self.counter.process_parallel(seqs);

        if !self.use_temporal {
            return;
        }

        let counts: Vec<(u64, u32)> = self
            .counter
            .counts
            .iter()
            .map(|entry| (*entry.key(), entry.value().load(Ordering::Relaxed)))
            .collect();

        self.temporal.rank_order_encode(&counts);
        self.simulate_temporal_spikes();
    }

    /// Simulate spikes with precise temporal control using public API
    fn simulate_temporal_spikes(&mut self) {
        use std::arch::x86_64::*;

        let steps = self.counter.get_steps();
        let pool_size = self.counter.get_pool_size();
        let threshold = self.counter.get_threshold();
        let leak = self.counter.get_leak();
        let spike_cost = self.counter.get_spike_cost();

        // Build time-to-neuron mapping
        let mut time_slots: Vec<Vec<usize>> = vec![Vec::new(); steps + 1];
        for (kmer, spike) in &self.temporal.spike_times {
            let neuron_idx = self.counter.map_kmer_to_neuron(*kmer);
            let time = spike.time_step as usize;
            if time <= steps && time > 0 {
                time_slots[time].push(neuron_idx);
            }
        }

        unsafe {
            let mut total_spikes: u64 = 0;

            for t in 1..=steps {
                let active_neurons = &time_slots[t];
                if active_neurons.is_empty() {
                    continue;
                }

                for batch_start in (0..active_neurons.len()).step_by(8) {
                    let batch_end = (batch_start + 8).min(active_neurons.len());
                    let batch_size = batch_end - batch_start;

                    let mut v_arr = [0.0f32; 8];
                    let t_arr = [threshold; 8];
                    let l_arr = [leak; 8];
                    let c_arr = [1.0f32; 8];

                    for (i, &neuron_idx) in
                        active_neurons[batch_start..batch_end].iter().enumerate()
                    {
                        v_arr[i] = self.counter.get_neuron_voltage(neuron_idx);
                    }

                    let v_vec = _mm256_loadu_ps(v_arr.as_ptr());
                    let t_vec = _mm256_loadu_ps(t_arr.as_ptr());
                    let l_vec = _mm256_loadu_ps(l_arr.as_ptr());
                    let c_vec = _mm256_loadu_ps(c_arr.as_ptr());

                    let v_new = _mm256_add_ps(_mm256_mul_ps(v_vec, l_vec), c_vec);
                    let spike_cmp = _mm256_cmp_ps(v_new, t_vec, _CMP_GE_OQ);
                    let spike_mask = _mm256_movemask_ps(spike_cmp) as u8;

                    let zero = _mm256_setzero_ps();
                    let v_final = _mm256_blendv_ps(v_new, zero, spike_cmp);
                    _mm256_storeu_ps(v_arr.as_mut_ptr(), v_final);

                    for (i, &neuron_idx) in
                        active_neurons[batch_start..batch_end].iter().enumerate()
                    {
                        self.counter.set_neuron_voltage(neuron_idx, v_arr[i]);
                        if (spike_mask >> i) & 1 == 1 {
                            self.counter.increment_neuron_spike(neuron_idx);
                            total_spikes += 1;
                        }
                    }
                }
            }

            self.counter.add_energy(total_spikes, spike_cost);
        }
    }

    /// Get temporal statistics
    pub fn temporal_stats(&self) -> TemporalStats {
        let mean_time = if self.temporal.spike_times.is_empty() {
            0.0
        } else {
            self.temporal
                .spike_times
                .values()
                .map(|s| s.time_step as f32)
                .sum::<f32>()
                / self.temporal.spike_times.len() as f32
        };

        TemporalStats {
            coefficient_of_variation: 0.0, // TODO: implement
            synchronous_groups: 0,         // TODO: implement
            total_temporal_spikes: self.temporal.spike_times.len(),
            mean_spike_time: mean_time,
        }
    }
}

#[derive(Debug)]
pub struct TemporalStats {
    pub coefficient_of_variation: f32,
    pub synchronous_groups: usize,
    pub total_temporal_spikes: usize,
    pub mean_spike_time: f32,
}
