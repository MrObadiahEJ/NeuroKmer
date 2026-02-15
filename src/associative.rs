// src/associative.rs
// Borrowed from classic Hopfield implementations + spiking-neural-networks weight patterns
// Willshaw (sparse, one-shot) – more memory-efficient than full Hopfield

use ndarray::{Array2, Array1};
use std::collections::{HashMap, HashSet};
use blake3;
use crate::NeuroResult;
use log::debug;


#[derive(Debug, Clone)]
pub struct WillshawNetwork {
    weights: Array2<u8>,  // Sparse binary weights
    pattern_size: usize,
    stored_count: usize,
}

impl WillshawNetwork {
    pub fn new(pattern_size: usize) -> Self {
        Self {
            weights: Array2::zeros((pattern_size, pattern_size)),
            pattern_size,
            stored_count: 0,
        }
    }

    /// Store a binary pattern (k-mer hashed to bits)
    pub fn store(&mut self, pattern: &[u8]) -> NeuroResult<()> {
        if pattern.len() != self.pattern_size {
            return Err("Pattern size mismatch".into());
        }
        for (i, &pi) in pattern.iter().enumerate() {
            for (j, &pj) in pattern.iter().enumerate() {
                if pi > 0 && pj > 0 {
                    self.weights[[i, j]] = 1;
                }
            }
        }
        self.stored_count += 1;
        debug!("Stored pattern {} (total: {})", self.stored_count, self.stored_count);
        Ok(())
    }

    /// Recall with noisy input – iterative convergence
    pub fn recall(&self, noisy: &[u8], steps: usize) -> NeuroResult<Vec<u8>> {
        if noisy.len() != self.pattern_size {
            return Err("Noisy pattern size mismatch".into());
        }
        let mut state = ndarray::Array1::from_vec(noisy.to_vec().iter().map(|&b| if b > 0 { 1i8 } else { 0 }).collect());
        for _ in 0..steps {
            let mut new_state = state.clone();
            for i in 0..self.pattern_size {
                let sum: i32 = self.weights.row(i).iter().zip(&state).map(|(&w, &s)| w as i32 * s as i32).sum();
                new_state[i] = if sum > 0 { 1 } else { 0 };
            }
            if new_state == state { break; }
            state = new_state;
        }
        Ok(state.mapv(|v| if v > 0 { 255u8 } else { 0 }).to_vec())
    }
}

pub struct KmerAssociativeMemory {
    willshaw: WillshawNetwork,
    kmer_to_pattern: HashMap<u64, Vec<u8>>,
    pattern_to_kmers: HashMap<Vec<u8>, HashSet<u64>>,
    pattern_size: usize,
}

impl KmerAssociativeMemory {
    pub fn new(k: usize) -> Self {
        // Pattern size = 2^k bits (for small k) or fixed size for large k
        let pattern_size = if k <= 10 { 1 << k } else { 1024 };
        Self {
            willshaw: WillshawNetwork::new(pattern_size),
            kmer_to_pattern: HashMap::new(),
            pattern_to_kmers: HashMap::new(),
            pattern_size,
        }
    }

    /// Convert k-mer to sparse binary pattern
    fn kmer_to_pattern(&self, kmer: u64) -> Vec<u8> {
        let mut pattern = vec![0; self.pattern_size];
        let hash = blake3::hash(&kmer.to_le_bytes());
        let bytes = hash.as_bytes();
        
        // Set ~1% of bits to 1 (sparse Willshaw pattern)
        for i in 0..(self.pattern_size / 100) {
            let idx = (bytes[i % 32] as usize) % self.pattern_size;
            pattern[idx] = 255;
        }
        pattern
    }

    /// Store a k-mer with its count
    pub fn store_kmer(&mut self, kmer: u64, count: u32) -> NeuroResult<()> {
        let pattern = self.kmer_to_pattern(kmer);
        
        // Store in Willshaw network
        self.willshaw.store(&pattern)?;
        
        // Update mappings
        self.kmer_to_pattern.insert(kmer, pattern.clone());
        self.pattern_to_kmers.entry(pattern).or_insert_with(HashSet::new).insert(kmer);
        
        Ok(())
    }

    /// Find similar k-mers given a noisy query
    pub fn find_similar(&self, query_kmer: u64, max_distance: usize) -> Vec<(u64, f32)> {
        let query_pattern = self.kmer_to_pattern(query_kmer);
        
        // Recall from Willshaw
        if let Ok(recalled) = self.willshaw.recall(&query_pattern, 10) {
            // Find which stored patterns are closest to recalled pattern
            let mut results = Vec::new();
            
            for (stored_pattern, kmers) in &self.pattern_to_kmers {
                let distance = self.hamming_distance(&recalled, stored_pattern);
                if distance <= max_distance {
                    let similarity = 1.0 - (distance as f32 / self.pattern_size as f32);
                    for &kmer in kmers {
                        results.push((kmer, similarity));
                    }
                }
            }
            
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            results
        } else {
            Vec::new()
        }
    }

    fn hamming_distance(&self, a: &[u8], b: &[u8]) -> usize {
        a.iter().zip(b).filter(|(x, y)| (*x > &0) != (*y > &0)).count()
    }
}