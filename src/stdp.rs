// src/stdp.rs
// Spike-Timing-Dependent Plasticity (STDP) for neuromorphic learning
// Based on Bi and Poo (1998) experimental results

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

use rand::RngExt;

/// STDP window: timing relationship between pre and post synaptic spikes
#[derive(Debug, Clone, Copy)]
pub struct STDPWindow {
    pub tau_plus: f32,   // Time constant for LTP (pre before post) - typically 20ms
    pub tau_minus: f32,  // Time constant for LTD (post before pre) - typically 20ms
    pub a_plus: f32,     // Maximum LTP amplitude
    pub a_minus: f32,    // Maximum LTD amplitude
    pub w_max: f32,      // Maximum synaptic weight
    pub w_min: f32,      // Minimum synaptic weight
}

impl Default for STDPWindow {
    fn default() -> Self {
        Self {
            tau_plus: 20.0,   // 20ms time constant
            tau_minus: 20.0,  // 20ms time constant
            a_plus: 0.01,     // 1% maximum potentiation
            a_minus: 0.0105,  // 1.05% maximum depression (slightly asymmetric)
            w_max: 1.0,       // Normalized weights 0-1
            w_min: 0.0,
        }
    }
}

/// Synaptic weight with history for STDP
#[derive(Debug)]
pub struct STDPWeight {
    pub weight: AtomicU32,  // Fixed-point: weight * 10000
    pub pre_spike_times: Vec<u32>,  // Recent pre-synaptic spike times
    pub post_spike_times: Vec<u32>, // Recent post-synaptic spike times
    pub last_update: u32,   // Last simulation step updated
}

impl STDPWeight {
    pub fn new(initial_weight: f32) -> Self {
        Self {
            weight: AtomicU32::new((initial_weight * 10000.0) as u32),
            pre_spike_times: Vec::with_capacity(100),
            post_spike_times: Vec::with_capacity(100),
            last_update: 0,
        }
    }

    pub fn get_weight(&self) -> f32 {
        self.weight.load(Ordering::Relaxed) as f32 / 10000.0
    }

    pub fn update_weight(&self, delta: f32, w_max: f32, w_min: f32) {
        let current = self.weight.load(Ordering::Relaxed) as f32 / 10000.0;
        let new_weight = (current + delta).clamp(w_min, w_max);
        self.weight.store((new_weight * 10000.0) as u32, Ordering::Relaxed);
    }
}

/// STDP synapse connecting pre and post synaptic neurons
#[derive(Debug)]
pub struct STDPSynapse {
    pub pre_neuron: usize,
    pub post_neuron: usize,
    pub weight: STDPWeight,
    pub delay: u32,  // Synaptic delay in time steps
}

/// STDP learning engine
pub struct STDPPlasticity {
    pub window: STDPWindow,
    pub synapses: HashMap<(usize, usize), STDPSynapse>, // (pre, post) -> synapse
    pub neuron_spike_times: HashMap<usize, Vec<u32>>,   // neuron_id -> spike times
    pub current_time: u32,
    pub learning_rate: f32,
}

impl STDPPlasticity {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            window: STDPWindow::default(),
            synapses: HashMap::new(),
            neuron_spike_times: HashMap::new(),
            current_time: 0,
            learning_rate,
        }
    }

    /// Add a synapse between two neurons
    pub fn add_synapse(&mut self, pre: usize, post: usize, initial_weight: f32, delay: u32) {
        let synapse = STDPSynapse {
            pre_neuron: pre,
            post_neuron: post,
            weight: STDPWeight::new(initial_weight),
            delay,
        };
        self.synapses.insert((pre, post), synapse);
    }

    /// Record a spike from a neuron
    pub fn record_spike(&mut self, neuron_id: usize, time_step: u32) {
        self.neuron_spike_times
            .entry(neuron_id)
            .or_default()
            .push(time_step);
        self.current_time = time_step;
    }

    /// Calculate STDP weight change based on spike timing
    /// delta_t = t_post - t_pre (positive = post after pre = LTP)
    pub fn calculate_stdp(&self, delta_t: f32) -> f32 {
        if delta_t > 0.0 {
            // LTP: pre before post (causal)
            self.window.a_plus * (-delta_t / self.window.tau_plus).exp()
        } else if delta_t < 0.0 {
            // LTD: post before pre (anti-causal)
            -self.window.a_minus * (delta_t / self.window.tau_minus).exp()
        } else {
            0.0 // No change if simultaneous
        }
    }

    /// Apply STDP learning to all synapses
    pub fn apply_learning(&mut self) {
        let synapse_keys: Vec<(usize, usize)> = self.synapses.keys().cloned().collect();
        
        for (pre, post) in synapse_keys {
            if let Some(synapse) = self.synapses.get(&(pre, post)) {
                let pre_times = self.neuron_spike_times.get(&pre).cloned().unwrap_or_default();
                let post_times = self.neuron_spike_times.get(&post).cloned().unwrap_or_default();
                
                let mut total_delta = 0.0;
                
                // Calculate all pairwise STDP contributions
                for &t_pre in &pre_times {
                    let t_pre_effective = t_pre + synapse.delay; // Account for synaptic delay
                    
                    for &t_post in &post_times {
                        let delta_t = t_post as f32 - t_pre_effective as f32;
                        
                        // Only consider spikes within STDP window (±100ms typically)
                        if delta_t.abs() < 100.0 {
                            let delta_w = self.calculate_stdp(delta_t);
                            total_delta += delta_w * self.learning_rate;
                        }
                    }
                }
                
                // Apply weight update
                if total_delta != 0.0 {
                    synapse.weight.update_weight(
                        total_delta,
                        self.window.w_max,
                        self.window.w_min
                    );
                }
            }
        }
    }

    /// Clear old spike history (call periodically to prevent memory growth)
    pub fn clear_history(&mut self, keep_last_ms: u32) {
        let cutoff = self.current_time.saturating_sub(keep_last_ms);
        
        for times in self.neuron_spike_times.values_mut() {
            times.retain(|&t| t >= cutoff);
        }
        
        // Also clear from weights
        for synapse in self.synapses.values_mut() {
            synapse.weight.pre_spike_times.retain(|&t| t >= cutoff);
            synapse.weight.post_spike_times.retain(|&t| t >= cutoff);
        }
    }

    /// Get weight statistics
    pub fn weight_stats(&self) -> WeightStats {
        let weights: Vec<f32> = self.synapses.values()
            .map(|s| s.weight.get_weight())
            .collect();
        
        if weights.is_empty() {
            return WeightStats::default();
        }
        
        let sum: f32 = weights.iter().sum();
        let mean = sum / weights.len() as f32;
        let variance = weights.iter()
            .map(|w| (w - mean).powi(2))
            .sum::<f32>() / weights.len() as f32;
        
        WeightStats {
            count: weights.len(),
            mean,
            std_dev: variance.sqrt(),
            min: *weights.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
            max: *weights.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
        }
    }
}

#[derive(Debug, Default)]
pub struct WeightStats {
    pub count: usize,
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
}

/// Integrate STDP with SpikingKmerCounter for learning k-mer patterns
pub struct STDPKmerLearner {
    pub counter: crate::spiking_hash::SpikingKmerCounter,
    pub stdp: STDPPlasticity,
    pub synapse_density: f32, // Probability of connection between neurons
    pub pattern_history: Vec<Vec<usize>>, // Recent k-mer patterns
}

impl STDPKmerLearner {
    pub fn new(
        k: usize,
        threshold: f32,
        leak: f32,
        refractory: u32,
        spike_cost: f64,
        pool_size: usize,
        use_canonical: bool,
        learning_rate: f32,
        synapse_density: f32,
    ) -> Self {
        let mut learner = Self {
            counter: crate::spiking_hash::SpikingKmerCounter::new(
                k, threshold, leak, refractory, spike_cost, pool_size, use_canonical
            ),
            stdp: STDPPlasticity::new(learning_rate),
            synapse_density,
            pattern_history: Vec::with_capacity(1000),
        };

        // Initialize random synapses between k-mer neurons
        learner.initialize_synapses(pool_size, synapse_density);
        learner
    }

    fn initialize_synapses(&mut self, pool_size: usize, density: f32) {
        use rand::Rng;
        let mut rng = rand::rng();
        
        let num_synapses = (pool_size as f32 * density) as usize;
        
        for _ in 0..num_synapses {
            let pre = rng.random_range(0..pool_size);
            let post = rng.random_range(0..pool_size);
            
            if pre != post {
                let weight = rng.random_range(0.1..0.5);
                let delay = rng.random_range(1..5); // 1-4 time steps delay
                self.stdp.add_synapse(pre, post, weight, delay);
            }
        }
        
        println!("Initialized {} STDP synapses", self.stdp.synapses.len());
    }

    /// Process sequence with STDP learning
    pub fn process_with_learning(&mut self, seqs: &[Vec<u8>]) {
        let k = self.counter.k;
        let use_canonical = self.counter.use_canonical;
        
        // Process each sequence
        for seq in seqs {
            if seq.len() < k {
                continue;
            }

            let mut active_neurons = Vec::new();
            
            // Extract k-mers and activate neurons
            if use_canonical {
                let mut rolling = crate::models::RollingKmerHash::new(k);
                rolling.init(&seq[0..k]);
                let canonical = rolling.canonical();
                let neuron = self.counter.map_kmer_to_neuron(canonical);
                active_neurons.push(neuron);
                
                for i in 1..=seq.len() - k {
                    let prev_base = seq[i - 1];
                    let next_base = seq[i + k - 1];
                    rolling.slide(next_base, prev_base);
                    let canonical = rolling.canonical();
                    let neuron = self.counter.map_kmer_to_neuron(canonical);
                    active_neurons.push(neuron);
                }
            } else {
                for window in seq.windows(k) {
                    let packed = crate::utils::pack_kmer(window);
                    let neuron = self.counter.map_kmer_to_neuron(packed);
                    active_neurons.push(neuron);
                }
            }

            // Simulate with temporal dynamics and record spikes
            self.simulate_and_learn(&active_neurons);
            self.pattern_history.push(active_neurons);
            
            // Keep history bounded
            if self.pattern_history.len() > 1000 {
                self.pattern_history.remove(0);
            }
        }

        // Apply STDP learning
        self.stdp.apply_learning();
        self.stdp.clear_history(1000); // Keep last 1000 time steps
    }

    fn simulate_and_learn(&mut self, active_neurons: &[usize]) {
        let steps = self.counter.get_steps();
        
        // Simple temporal coding: earlier spikes for repeated patterns
        for (t, &neuron) in active_neurons.iter().enumerate().take(steps) {
            let time_step = t as u32 + 1;
            
            // Record spike for STDP
            self.stdp.record_spike(neuron, time_step);
            
            // Update neuron state (simplified)
            if let Some(n) = self.counter.neurons_mut().get_mut(neuron) {
                n.voltage = 0.0; // Reset after spike
                n.spike_count += 1;
            }
        }
    }

    /// Recall/learned pattern completion
    pub fn pattern_completion(&self, seed_neurons: &[usize], steps: usize) -> Vec<usize> {
        let mut active = seed_neurons.to_vec();
        let mut completed = seed_neurons.to_vec();
        
        for _ in 0..steps {
            let mut next_active = Vec::new();
            
            for &pre in &active {
                // Find synapses from pre-synaptic neurons
                for ((p, post), synapse) in &self.stdp.synapses {
                    if *p == pre && synapse.weight.get_weight() > 0.3 {
                        // Strong synapse activates post-synaptic neuron
                        if !completed.contains(post) {
                            next_active.push(*post);
                            completed.push(*post);
                        }
                    }
                }
            }
            
            if next_active.is_empty() {
                break;
            }
            active = next_active;
        }
        
        completed
    }

    /// Get learning statistics
    pub fn learning_stats(&self) -> LearningStats {
        let weight_stats = self.stdp.weight_stats();
        let pattern_count = self.pattern_history.len();
        
        // Calculate pattern diversity
        let unique_patterns: std::collections::HashSet<_> = self.pattern_history.iter().cloned().collect();
        
        LearningStats {
            synapse_count: weight_stats.count,
            mean_weight: weight_stats.mean,
            weight_std: weight_stats.std_dev,
            patterns_learned: pattern_count,
            pattern_diversity: unique_patterns.len(),
        }
    }
}

#[derive(Debug)]
pub struct LearningStats {
    pub synapse_count: usize,
    pub mean_weight: f32,
    pub weight_std: f32,
    pub patterns_learned: usize,
    pub pattern_diversity: usize,
}