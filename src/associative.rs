// src/associative.rs
// Borrowed from classic Hopfield implementations + spiking-neural-networks weight patterns
// Willshaw (sparse, one-shot) – more memory-efficient than full Hopfield

use ndarray::Array2;
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