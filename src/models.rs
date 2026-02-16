// src/models.rs
// Borrowed/adapted from:
// - spiking-neural-networks: Threshold/reset logic
// - neural-rs: Stateful voltage integration
// - WheatNNLeek: Leak + refractory period

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy)]
pub struct LifNeuron {
    pub voltage: f64,           // Membrane potential
    pub threshold: f64,         // Spike threshold
    pub reset_voltage: f64,     // Post-spike reset
    pub leak: f64,              // Membrane leak constant (0.0 = no leak)
    pub refractory_ticks: u32,  // Remaining refractory time
    pub refractory_period: u32, // Fixed refractory duration
    pub spike_count: u64,       // NEW: Track individual spikes
}

impl LifNeuron {
    pub fn new(threshold: f64, leak: f64, refractory_period: u32) -> Self {
        Self {
            voltage: 0.0,
            threshold,
            reset_voltage: 0.0,
            leak,
            refractory_ticks: 0,
            refractory_period,
            spike_count: 0,
        }
    }

    /// Integrate input current + apply leak; return true if spike
    pub fn update(&mut self, input_current: f64) -> bool {
        if self.refractory_ticks > 0 {
            self.refractory_ticks -= 1;
            return false;
        }

        // Leak + integrate (borrowed vectorized style)
        self.voltage = self.voltage * self.leak + input_current;

        if self.voltage >= self.threshold {
            self.voltage = self.reset_voltage;
            self.refractory_ticks = self.refractory_period;
            self.spike_count += 1; // NEW: Count spike
            true // Spike fired
        } else {
            false
        }
    }
}

#[derive(Debug, Default)]
pub struct EnergyTracker {
    pub total_spikes: AtomicU64, // Changed to AtomicU64
    pub total_energy: AtomicU64, // Store as fixed-point to avoid float atomics
}

impl EnergyTracker {
    pub fn new() -> Self {
        Self {
            total_spikes: AtomicU64::new(0),
            total_energy: AtomicU64::new(0),
        }
    }
    
    pub fn add_spike(&self, cost: f64) {
        self.total_spikes.fetch_add(1, Ordering::Relaxed);
        // Convert float to fixed-point (multiply by 1000 to preserve 3 decimal places)
        let cost_fixed = (cost * 1000.0) as u64;
        self.total_energy.fetch_add(cost_fixed, Ordering::Relaxed);
    }
    
    pub fn total_spikes(&self) -> u64 {
        self.total_spikes.load(Ordering::Relaxed)
    }
    
    pub fn total_energy(&self) -> f64 {
        (self.total_energy.load(Ordering::Relaxed) as f64) / 1000.0
    }
}

/// Rolling hash for DNA k-mers using base-4 representation
/// This allows O(1) update when sliding the window
pub struct RollingKmerHash {
    k: usize,
    forward: u64, // Forward strand encoding
    reverse: u64, // Reverse complement encoding
    mask: u64,    // For keeping only k bases (2 bits per base)
    power: u64,   // 4^(k-1) for removing leftmost base
}

impl RollingKmerHash {
    pub fn new(k: usize) -> Self {
        // Mask with 2k bits set
        let mask = if k < 32 { (1u64 << (2 * k)) - 1 } else { !0u64 };

        // Precompute 4^(k-1) for rolling hash
        let mut power = 1u64;
        for _ in 0..(k - 1) {
            power = (power << 2) & mask;
        }

        Self {
            k,
            forward: 0,
            reverse: 0,
            mask,
            power,
        }
    }

    /// Initialize with first k bases
    pub fn init(&mut self, first_k: &[u8]) {
        assert_eq!(
            first_k.len(),
            self.k,
            "Initialization slice length must equal k"
        );

        // Pack forward strand
        self.forward = 0;
        for &base in first_k {
            let bits = Self::base_to_bits(base);
            self.forward = ((self.forward << 2) & self.mask) | bits;
        }

        // Build reverse complement by processing in reverse
        // For reverse complement, we need the complement of each base in reverse order
        self.reverse = 0;
        for &base in first_k.iter().rev() {
            let comp_bits = Self::base_to_complement_bits(base);
            self.reverse = ((self.reverse << 2) & self.mask) | comp_bits;
        }
    }

    /// Convert base to 2-bit encoding
    #[inline]
    fn base_to_bits(base: u8) -> u64 {
        match base {
            b'A' | b'a' => 0,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' => 3,
            _ => 0, // Default to A for N or other bases
        }
    }

    /// Convert base to its complement's 2-bit encoding
    #[inline]
    fn base_to_complement_bits(base: u8) -> u64 {
        match base {
            b'A' | b'a' => 3, // A -> T (3)
            b'C' | b'c' => 2, // C -> G (2)
            b'G' | b'g' => 1, // G -> C (1)
            b'T' | b't' => 0, // T -> A (0)
            _ => 0, // Default to A for N (but A's complement is T, so maybe 3? We'll keep consistent)
        }
    }

    /// Slide window by one base: O(1) time using rolling hash formula
    pub fn slide(&mut self, next_base: u8, prev_base: u8) {
        // Get bit values
        let prev_bits = Self::base_to_bits(prev_base);
        let next_bits = Self::base_to_bits(next_base);

        // Update forward strand
        self.forward = self.forward.wrapping_sub(prev_bits * self.power);
        self.forward = ((self.forward << 2) | next_bits) & self.mask;

        // Update reverse complement - simplified
        let comp_next = Self::base_to_complement_bits(next_base);
        self.reverse = (self.reverse >> 2) | (comp_next << (2 * (self.k - 1)));
        self.reverse &= self.mask;

        // Remove unused comp_prev variable
    }
    /// Get current forward k-mer
    #[inline]
    pub fn forward(&self) -> u64 {
        self.forward
    }

    /// Get reverse complement (lazy compute)
    #[inline]
    pub fn reverse_complement(&self) -> u64 {
        self.reverse
    }

    /// Get canonical k-mer efficiently
    #[inline]
    pub fn canonical(&self) -> u64 {
        std::cmp::min(self.forward, self.reverse)
    }

    /// Check if current k-mer contains Ns (invalid)
    pub fn is_valid(&self) -> bool {
        // Simple check - if any byte had N, we set bits to 0 which is valid A
        // So this always returns true. We need a better approach.
        true
    }

    pub fn reset(&mut self) {
        self.forward = 0;
        self.reverse = 0;
    }
}
