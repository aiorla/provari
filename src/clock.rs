//! The search's clock: the standard one, but for WebAssembly in a browser,
//! which has none of its own, the page's (`performance.now()`), imported.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

#[cfg(target_arch = "wasm32")]
pub use web::Instant;

#[cfg(target_arch = "wasm32")]
mod web {
    use std::ops::{Add, Sub};
    use std::time::Duration;

    #[link(wasm_import_module = "provari")]
    unsafe extern "C" {
        /// Milliseconds on the page's monotonic clock.
        fn now_ms() -> f64;
    }

    /// A moment, in milliseconds on the page's clock.
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct Instant(f64);

    impl Instant {
        pub fn now() -> Instant {
            // SAFETY: the page provides the import, which takes nothing.
            Instant(unsafe { now_ms() })
        }

        pub fn duration_since(&self, earlier: Instant) -> Duration {
            Duration::from_secs_f64((self.0 - earlier.0).max(0.0) / 1000.0)
        }

        pub fn elapsed(&self) -> Duration {
            Instant::now().duration_since(*self)
        }
    }

    impl Add<Duration> for Instant {
        type Output = Instant;

        fn add(self, duration: Duration) -> Instant {
            Instant(self.0 + duration.as_secs_f64() * 1000.0)
        }
    }

    impl Sub<Instant> for Instant {
        type Output = Duration;

        fn sub(self, earlier: Instant) -> Duration {
            self.duration_since(earlier)
        }
    }
}
