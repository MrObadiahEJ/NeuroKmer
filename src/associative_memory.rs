// src/associative_memory.rs - FIXED with sparse representation
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicI32, Ordering};

/// Sparse associative memory using local connections
#[derive(Debug)]
pub struct AssociativeMemory {
    pub pattern_size: usize,
    pub stored_patterns: Vec<Vec<usize>>,
    pub pattern_strengths: Vec<f32>,
    // SPARSE: Only store non-zero weights as HashMap (i, j) -> weight
    pub sparse_weights: HashMap<(usize, usize), AtomicI32>, // Fixed-point: weight * 1000
    pub max_capacity: usize,
    pub connectivity: f32, // Fraction of possible connections actually stored
}

impl AssociativeMemory {
    pub fn new(pattern_size: usize, max_capacity: usize) -> Self {
        let actual_capacity = max_capacity.min((pattern_size as f32 * 0.14) as usize);
        
        Self {
            pattern_size,
            stored_patterns: Vec::with_capacity(actual_capacity),
            pattern_strengths: Vec::with_capacity(actual_capacity),
            sparse_weights: HashMap::new(),
            max_capacity: actual_capacity,
            connectivity: 0.1, // 10% sparse connectivity
        }
    }

    pub fn store_pattern(&mut self, pattern: &[usize]) -> Result<(), &'static str> {
        if self.stored_patterns.len() >= self.max_capacity {
            return Err("Memory capacity exceeded");
        }

        // Hebbian learning on sparse connections
        // Only connect neurons that are close in the pattern (local connectivity)
        let local_radius = (self.pattern_size as f32 * self.connectivity) as usize;

        for (i, &pre) in pattern.iter().enumerate() {
            for (j, &post) in pattern.iter().enumerate() {
                if pre != post {
                    let distance = if pre > post { pre - post } else { post - pre };
                    
                    // Local connectivity: only connect nearby neurons
                    if distance <= local_radius || distance >= self.pattern_size - local_radius {
                        let key = (pre.min(post), post.max(post));
                        let delta = 1000; // LTP: +1.0 in fixed-point
                        
                        self.sparse_weights
                            .entry(key)
                            .or_insert_with(|| AtomicI32::new(0))
                            .fetch_add(delta, Ordering::Relaxed);
                    }
                }
            }
        }

        self.stored_patterns.push(pattern.to_vec());
        self.pattern_strengths.push(1.0);
        
        Ok(())
    }

    pub fn recall(&self, cue: &[usize], max_iterations: usize, threshold: f32) -> (Vec<usize>, f32) {
        let mut state = vec![0.0f32; self.pattern_size];
        for &idx in cue {
            if idx < self.pattern_size {
                state[idx] = 1.0;
            }
        }

        let mut prev_state = state.clone();
        let mut converged = false;
        let mut iterations = 0;

        while !converged && iterations < max_iterations {
            let mut new_state = state.clone();
            
            for i in 0..self.pattern_size {
                // Sparse calculation: only check connected neurons
                let mut h = 0.0f32;
                
                // Check all weights involving neuron i
                for j in 0..self.pattern_size {
                    if i != j {
                        let key = (i.min(j), i.max(j));
                        if let Some(weight) = self.sparse_weights.get(&key) {
                            let w = weight.load(Ordering::Relaxed) as f32 / 1000.0;
                            h += w * state[j];
                        }
                    }
                }
                
                new_state[i] = if h > threshold { 1.0 } else { -1.0 };
            }

            let diff: f32 = new_state.iter().zip(&prev_state)
                .map(|(a, b)| (a - b).abs())
                .sum();
            
            converged = diff < 0.001 * self.pattern_size as f32;
            prev_state = new_state.clone();
            state = new_state;
            iterations += 1;
        }

        let recalled_pattern: Vec<usize> = state.iter()
            .enumerate()
            .filter(|&(_, &v)| v > 0.0)
            .map(|(i, _)| i)
            .collect();

        let confidence = if converged {
            1.0 - (iterations as f32 / max_iterations as f32)
        } else {
            0.5
        };

        (recalled_pattern, confidence)
    }

    pub fn content_addressable_search(&self, query: &[usize], top_k: usize) -> Vec<(usize, f32)> {
        let mut similarities: Vec<(usize, f32)> = Vec::with_capacity(self.stored_patterns.len());
        
        for (idx, pattern) in self.stored_patterns.iter().enumerate() {
            let overlap: f32 = query.iter()
                .filter(|&&q| pattern.contains(&q))
                .count() as f32;
            let union: HashSet<usize> = query.iter().copied()
                .chain(pattern.iter().copied())
                .collect();
            let jaccard = overlap / union.len().max(1) as f32;
            similarities.push((idx, jaccard));
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.into_iter().take(top_k).collect()
    }

    pub fn sparsify(&self, dense_input: &[f32], target_sparsity: f32) -> Vec<usize> {
        let k = (self.pattern_size as f32 * target_sparsity) as usize;
        
        let mut activated: Vec<(usize, f32)> = dense_input.iter()
            .enumerate()
            .filter(|&(_, &v)| v > 0.0)
            .map(|(i, &v)| (i, v))
            .collect();
        
        activated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let mut winners = Vec::new();
        let inhibition_radius = 5;

        for (idx, _strength) in activated {
            let too_close = winners.iter().any(|&w| (w as i32 - idx as i32).abs() <= inhibition_radius);
            if !too_close {
                winners.push(idx);
                if winners.len() >= k {
                    break;
                }
            }
        }
        
        winners
    }
}

/// Semantic k-mer query system
#[derive(Debug)]
pub struct SemanticKmerQuery {
    pub memory: AssociativeMemory,
    pub kmer_to_neuron: HashMap<u64, usize>,
    pub neuron_to_kmer: HashMap<usize, Vec<u64>>,
    pub k: usize,
    pub neuron_pool_size: usize, // Actual pool size for mapping
}

impl SemanticKmerQuery {
    pub fn new(k: usize, pool_size: usize) -> Self {
        // Use sqrt(pool_size) for associative memory to keep it sparse
        // For 100k pool, use ~1000 neurons for Hopfield memory
        let memory_size = ((pool_size as f32).sqrt() as usize * 3).min(2000).max(100);
        
        Self {
            memory: AssociativeMemory::new(memory_size, memory_size / 5),
            kmer_to_neuron: HashMap::new(),
            neuron_to_kmer: HashMap::new(),
            k,
            neuron_pool_size: pool_size,
        }
    }

    pub fn index_sequences(&mut self, seqs: &[Vec<u8>]) {
        let k = self.k;
        
        for seq in seqs {
            if seq.len() < k {
                continue;
            }

            let mut pattern = Vec::new();
            
            for window in seq.windows(k) {
                let packed = crate::utils::pack_kmer(window);
                // Map to neuron within memory.pattern_size range
                let neuron_idx = self.map_kmer_to_neuron(packed) % self.memory.pattern_size;
                
                self.kmer_to_neuron.insert(packed, neuron_idx);
                self.neuron_to_kmer.entry(neuron_idx).or_default().push(packed);
                pattern.push(neuron_idx);
            }

            pattern.sort_unstable();
            pattern.dedup();
            
            if !pattern.is_empty() && pattern.len() <= self.memory.pattern_size / 2 {
                let _ = self.memory.store_pattern(&pattern);
            }
        }
    }

    fn map_kmer_to_neuron(&self, packed: u64) -> usize {
        use std::hash::{Hash, Hasher};
        use siphasher::sip::SipHasher13;
        
        let mut hasher = SipHasher13::new_with_keys(0, 0);
        packed.hash(&mut hasher);
        (hasher.finish() % self.neuron_pool_size as u64) as usize
    }

    pub fn query(&self, query_kmers: &[u64], top_k: usize) -> Vec<SemanticMatch> {
        let query_pattern: Vec<usize> = query_kmers.iter()
            .filter_map(|&kmer| self.kmer_to_neuron.get(&kmer).copied())
            .collect();

        if query_pattern.is_empty() {
            return Vec::new();
        }

        let matches = self.memory.content_addressable_search(&query_pattern, top_k);
        
        matches.into_iter().map(|(pattern_idx, similarity)| {
            let stored_pattern = &self.memory.stored_patterns[pattern_idx];
            
            let matched_kmers: Vec<u64> = stored_pattern.iter()
                .flat_map(|&neuron| {
                    self.neuron_to_kmer.get(&neuron)
                        .cloned()
                        .unwrap_or_default()
                })
                .collect();

            SemanticMatch {
                pattern_index: pattern_idx,
                similarity_score: similarity,
                matched_kmers,
                confidence: self.memory.pattern_strengths.get(pattern_idx).copied().unwrap_or(0.5),
            }
        }).collect()
    }

    pub fn fuzzy_query(&self, query_kmer: u64, _max_hamming: u32) -> Vec<(u64, f32)> {
        let query_neuron = match self.kmer_to_neuron.get(&query_kmer) {
            Some(&n) => n,
            None => return Vec::new(),
        };

        let mut coactivation_scores: HashMap<u64, f32> = HashMap::new();
        
        for pattern in &self.memory.stored_patterns {
            if pattern.contains(&query_neuron) {
                for &neuron in pattern {
                    if neuron != query_neuron {
                        if let Some(kmers) = self.neuron_to_kmer.get(&neuron) {
                            for &kmer in kmers {
                                *coactivation_scores.entry(kmer).or_default() += 1.0;
                            }
                        }
                    }
                }
            }
        }

        let total_patterns = self.memory.stored_patterns.len().max(1) as f32;
        let mut results: Vec<(u64, f32)> = coactivation_scores.into_iter()
            .map(|(kmer, score)| (kmer, score / total_patterns))
            .filter(|(_, score)| *score > 0.1)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(10);
        results
    }

    pub fn complete_pattern(&self, partial: &[u64]) -> Vec<u64> {
        let cue: Vec<usize> = partial.iter()
            .filter_map(|&kmer| self.kmer_to_neuron.get(&kmer).copied())
            .collect();

        if cue.is_empty() {
            return Vec::new();
        }

        let (recalled, _confidence) = self.memory.recall(&cue, 50, 0.0);
        
        recalled.into_iter()
            .flat_map(|neuron| self.neuron_to_kmer.get(&neuron).cloned().unwrap_or_default())
            .collect()
    }

    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            patterns_stored: self.memory.stored_patterns.len(),
            capacity_used: self.memory.stored_patterns.len() as f32 / self.memory.max_capacity as f32,
            unique_kmers_indexed: self.kmer_to_neuron.len(),
            mean_pattern_size: if self.memory.stored_patterns.is_empty() {
                0.0
            } else {
                self.memory.stored_patterns.iter()
                    .map(|p| p.len() as f32)
                    .sum::<f32>() / self.memory.stored_patterns.len() as f32
            },
            sparse_connections: self.memory.sparse_weights.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SemanticMatch {
    pub pattern_index: usize,
    pub similarity_score: f32,
    pub matched_kmers: Vec<u64>,
    pub confidence: f32,
}

#[derive(Debug)]
pub struct MemoryStats {
    pub patterns_stored: usize,
    pub capacity_used: f32,
    pub unique_kmers_indexed: usize,
    pub mean_pattern_size: f32,
    pub sparse_connections: usize,
}

/// Integrated system: STDP + Associative Memory
pub struct NeuroAssociativeSystem {
    pub stdp_learner: crate::stdp::STDPKmerLearner,
    pub semantic_memory: SemanticKmerQuery,
    pub integration_threshold: f32,
}

impl NeuroAssociativeSystem {
    pub fn new(
        k: usize,
        pool_size: usize,
        learning_rate: f32,
        synapse_density: f32,
    ) -> Self {
        Self {
            stdp_learner: crate::stdp::STDPKmerLearner::new(
                k, 1.0, 0.95, 2, 1.0, pool_size, true, learning_rate, synapse_density
            ),
            semantic_memory: SemanticKmerQuery::new(k, pool_size),
            integration_threshold: 0.5,
        }
    }

    pub fn learn(&mut self, seqs: &[Vec<u8>]) {
        self.stdp_learner.process_with_learning(seqs);
        self.semantic_memory.index_sequences(seqs);
    }

    pub fn intelligent_query(&self, query: &[u64]) -> IntelligentResults {
        let semantic_matches = self.semantic_memory.query(query, 5);
        
        let seed_neurons: Vec<usize> = query.iter()
            .filter_map(|&k| self.semantic_memory.kmer_to_neuron.get(&k).copied())
            .collect();
        let stdp_completed = self.stdp_learner.pattern_completion(&seed_neurons, 3);
        
        let stdp_kmers: Vec<u64> = stdp_completed.iter()
            .flat_map(|&neuron| {
                self.semantic_memory.neuron_to_kmer.get(&neuron)
                    .cloned()
                    .unwrap_or_default()
            })
            .collect();

        IntelligentResults {
            semantic_matches,
            stdp_predictions: stdp_kmers,
            combined_confidence: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct IntelligentResults {
    pub semantic_matches: Vec<SemanticMatch>,
    pub stdp_predictions: Vec<u64>,
    pub combined_confidence: f32,
}