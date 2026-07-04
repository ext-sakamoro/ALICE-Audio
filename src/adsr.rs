//! `Adsr` — ADSR envelope generator.

/// ADSR envelope generator. Times are in samples.
#[derive(Debug, Clone, Copy)]
pub struct Adsr {
    pub attack: usize,
    pub decay: usize,
    pub sustain_level: f64,
    pub release: usize,
}

impl Adsr {
    pub const fn new(attack: usize, decay: usize, sustain_level: f64, release: usize) -> Self {
        Self {
            attack,
            decay,
            sustain_level,
            release,
        }
    }

    /// Generate the full envelope for a note of `hold_samples` duration
    /// (attack + decay + sustain hold + release).
    pub fn generate(&self, hold_samples: usize) -> Vec<f64> {
        let total = hold_samples + self.release;
        let mut env = Vec::with_capacity(total);
        for i in 0..total {
            env.push(self.sample(i, hold_samples));
        }
        env
    }

    /// Get envelope value at sample index `i` with note-off at `hold_samples`.
    pub fn sample(&self, i: usize, hold_samples: usize) -> f64 {
        if i < self.attack {
            i as f64 / self.attack.max(1) as f64
        } else if i < self.attack + self.decay {
            let t = (i - self.attack) as f64 / self.decay.max(1) as f64;
            1.0 - t * (1.0 - self.sustain_level)
        } else if i < hold_samples {
            self.sustain_level
        } else if i < hold_samples + self.release {
            let t = (i - hold_samples) as f64 / self.release.max(1) as f64;
            self.sustain_level * (1.0 - t)
        } else {
            0.0
        }
    }
}
