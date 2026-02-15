// src/models.rs
// Borrowed/adapted from:
// - spiking-neural-networks: Threshold/reset logic
// - neural-rs: Stateful voltage integration
// - WheatNNLeek: Leak + refractory period

#[derive(Debug, Clone, Copy)]
pub struct LifNeuron {
    pub voltage: f64,       // Membrane potential
    pub threshold: f64,     // Spike threshold
    pub reset_voltage: f64, // Post-spike reset
    pub leak: f64,          // Membrane leak constant (0.0 = no leak)
    pub refractory_ticks: u32, // Remaining refractory time
    pub refractory_period: u32, // Fixed refractory duration
    pub spike_count: u64,  // NEW: Track individual spikes
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
            self.spike_count += 1;  // NEW: Count spike
            true  // Spike fired
        } else {
            false
        }
    }
}

#[derive(Debug, Default)]
pub struct EnergyTracker {
    pub total_spikes: u64,
    pub total_energy: f64,  // Simulated "energy" (e.g., spikes * cost)
}