//! Lets one of the threads that search a position together change what they
//! share while the others wait: it asks them to wait, each does at its next
//! node, outside any change of its own, and once all of them wait, it makes
//! its change and lets them go on. The search drops its mates in one, and
//! rebuilds or drops its table's filter, this way, as these move what other
//! threads read.

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};

pub struct Pause {
    /// Whether a thread waits for the others to: read at every node.
    asked: AtomicBool,
    state: Mutex<State>,
    changed: Condvar,
}

#[derive(Default)]
struct State {
    /// The threads searching, which a change waits for.
    members: usize,
    /// The members waiting for a change to end.
    waiting: usize,
    /// Whether a thread makes a change.
    held: bool,
    /// The changes made, so that a waiting thread knows when its ends.
    changes: u64,
}

impl Default for Pause {
    fn default() -> Pause {
        Pause::new()
    }
}

impl Pause {
    pub fn new() -> Pause {
        Pause {
            asked: AtomicBool::new(false),
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
        }
    }

    /// Counts a thread among those a change waits for, until it leaves.
    pub fn enter(&self) {
        self.lock().members += 1;
    }

    /// Counts a thread out, as it stops searching.
    pub fn leave(&self) {
        self.lock().members -= 1;
        self.changed.notify_all();
    }

    /// Waits while another thread makes a change, at a node; returns whether
    /// it was asked to, so that the thread takes again what a change moves.
    #[inline(always)]
    pub fn check(&self) -> bool {
        if self.asked.load(Relaxed) {
            self.wait();
            return true;
        }
        false
    }

    #[cold]
    fn wait(&self) {
        let state = self.lock();
        if state.held {
            drop(self.wait_out(state));
        }
    }

    /// Waits as a member until the change being made ends.
    fn wait_out<'a>(&self, mut state: MutexGuard<'a, State>) -> MutexGuard<'a, State> {
        let change = state.changes;
        state.waiting += 1;
        self.changed.notify_all();
        while state.changes == change {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.waiting -= 1;
        state
    }

    /// Makes `change` while every other member waits. A member that also
    /// asks waits for this change first, and makes its own after. The lock
    /// stays held throughout, so that a thread let go by the last change
    /// cannot go on before this one ends.
    pub fn alone<R>(&self, change: impl FnOnce() -> R) -> R {
        let mut state = self.lock();
        while state.held {
            state = self.wait_out(state);
        }
        state.held = true;
        self.asked.store(true, Relaxed);
        while state.waiting + 1 < state.members {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        // Ends the change even if it panics, so that the others go on.
        struct End<'a, 'b> {
            pause: &'a Pause,
            state: MutexGuard<'b, State>,
        }
        impl Drop for End<'_, '_> {
            fn drop(&mut self) {
                self.state.held = false;
                self.state.changes += 1;
                self.pause.asked.store(false, Relaxed);
                self.pause.changed.notify_all();
            }
        }
        let _end = End { pause: self, state };
        change()
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

    use super::Pause;

    /// Threads that each add to a count of their own while now and then one
    /// of them reads every count twice alone: no count moves between its
    /// two reads, whichever thread asks and however often.
    #[test]
    fn a_change_is_made_alone() {
        let pause = Pause::new();
        let threads = 4;
        let counts: Vec<AtomicU64> = (0..threads).map(|_| AtomicU64::new(0)).collect();
        let changes = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for thread in 0..threads {
                pause.enter();
                let (pause, counts, changes) = (&pause, &counts, &changes);
                scope.spawn(move || {
                    for step in 0..20_000u64 {
                        pause.check();
                        counts[thread].fetch_add(1, Relaxed);
                        if (step + thread as u64).is_multiple_of(1000) {
                            pause.alone(|| {
                                let before: Vec<u64> =
                                    counts.iter().map(|count| count.load(Relaxed)).collect();
                                std::thread::yield_now();
                                let after: Vec<u64> =
                                    counts.iter().map(|count| count.load(Relaxed)).collect();
                                assert_eq!(before, after);
                                changes.fetch_add(1, Relaxed);
                            });
                        }
                    }
                    pause.leave();
                });
            }
        });
        assert_eq!(changes.load(Relaxed), threads * 20);
        assert!(counts.iter().all(|count| count.load(Relaxed) == 20_000));
    }

    /// A change asked for by a thread that is not searching, or with every
    /// other thread gone, is made at once.
    #[test]
    fn a_change_waits_for_members_only() {
        let pause = Pause::new();
        assert_eq!(pause.alone(|| 7), 7);
        pause.enter();
        pause.enter();
        pause.leave();
        assert_eq!(pause.alone(|| 8), 8);
        pause.check();
    }
}
