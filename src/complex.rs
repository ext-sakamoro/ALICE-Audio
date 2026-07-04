//! Complex number + FFT (Cooley-Tukey radix-2 DIT).

use core::f64::consts::PI;

/// Minimal complex number for FFT.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn magnitude(self) -> f64 {
        self.re.hypot(self.im)
    }

    #[inline]
    pub fn phase(self) -> f64 {
        self.im.atan2(self.re)
    }
}

impl core::ops::Add for Complex {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl core::ops::Sub for Complex {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl core::ops::Mul for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re.mul_add(rhs.re, -(self.im * rhs.im)),
            self.re.mul_add(rhs.im, self.im * rhs.re),
        )
    }
}

/// Compute the FFT of `data` in-place. Length must be a power of two.
///
/// # Panics
///
/// Panics if `data.len()` is not a power of two.
pub fn fft(data: &mut [Complex]) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT length must be power of two");
    if n <= 1 {
        return;
    }
    bit_reverse_permutation(data);
    let mut size = 2;
    while size <= n {
        let half = size / 2;
        let angle_step = -2.0 * PI / size as f64;
        for k in 0..half {
            let w = Complex::new((angle_step * k as f64).cos(), (angle_step * k as f64).sin());
            let mut j = k;
            while j < n {
                let t = w * data[j + half];
                let u = data[j];
                data[j] = u + t;
                data[j + half] = u - t;
                j += size;
            }
        }
        size *= 2;
    }
}

/// Compute the inverse FFT in-place.
///
/// # Panics
///
/// Panics if `data.len()` is not a power of two.
pub fn ifft(data: &mut [Complex]) {
    let n = data.len();
    for c in data.iter_mut() {
        c.im = -c.im;
    }
    fft(data);
    let inv = 1.0 / n as f64;
    for c in data.iter_mut() {
        c.re *= inv;
        c.im = -c.im * inv;
    }
}

fn bit_reverse_permutation(data: &mut [Complex]) {
    let n = data.len();
    let bits = n.trailing_zeros();
    for i in 0..n {
        let rev = i.reverse_bits() >> (usize::BITS - bits);
        if i < rev {
            data.swap(i, rev);
        }
    }
}
