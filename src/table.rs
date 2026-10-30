//! The search's bounded memory: a fixed-size table of proof and disproof
//! numbers, in clusters of four entries that share a cache line, behind a
//! filter of the positions it may hold while the filter is sparse. A new
//! entry replaces the one whose search took the least work, the cheapest to
//! redo.
//!
//! Threads may share a table (`shared`). An entry is two words, written one
//! after the other, so a thread can read one word of an entry that another
//! is writing; shared, an entry's check bits are kept mixed with its second
//! word, so that a slot read half old and half new reads as another
//! position's, a miss. The filter's bits are then set by atomic operations,
//! so that none is lost, and not counted, which would have the threads wait
//! on one another for the count: they look at a sample of the filter now and
//! then instead (`filter_is_dense`).
//!
//! Whether the filter is read changes only in `note`, `put` and
//! `rebuild_filter` for a table of its own, which return it, and, while the
//! threads that share one wait, in `drop_filter` and `rebuild_filter`: a
//! thread takes it again from `filtering` after those, and passes it to the
//! reads. A flag read at every lookup would cost threads in WebKit, which
//! compiles WebAssembly's atomic reads as atomic adds of zero: each read takes
//! the flag's cache line from the other threads' caches, as a write does.

use std::collections::TryReserveError;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};

/// What the table knows about a position.
#[derive(Clone, Copy)]
pub enum Known {
    Open {
        proof: u32,
        disproof: u32,
    },
    /// The attacker cannot win from here: on any line, or only while the
    /// position with these check bits is on the line, and from this 50-move
    /// counter upward.
    Refuted {
        cycle: Option<u32>,
        from: u32,
    },
    /// The attacker mates at once with the move of these bits, believed only
    /// once checked: another position can share the entry.
    Mate(u16),
}

/// Bits of a hash that tell positions in the same cluster apart; never zero,
/// which marks an empty slot.
pub fn check(hash: u64) -> u32 {
    ((hash >> 32) as u32).max(1)
}

/// One entry, in two words: the check bits and the first number, then the
/// disproof number and the meta, the work behind the entry, saturating,
/// above the refutation's counter. The first number is the proof number, for
/// a refutation (disproof zero) the repeated position's check bits, and for a
/// mate in one zero, with its move's bits as the disproof number.
#[derive(Default)]
#[repr(C)]
struct Slot {
    head: AtomicU64,
    tail: AtomicU64,
}

const WORK_MAX: u64 = (1 << 24) - 1;

/// The work behind an entry, from its second word.
fn work_of(tail: u64) -> u32 {
    (tail >> 40) as u32
}

/// What an entry's second word mixes into its check bits when threads share
/// the table.
fn mix(tail: u64) -> u32 {
    tail as u32 ^ (tail >> 32) as u32
}

#[derive(Default)]
#[repr(C, align(64))]
struct Cluster {
    slots: [Slot; 4],
}

pub struct Table {
    clusters: Vec<Cluster>,
    /// The bits `index` turns a hash's low half by: as many as the clusters
    /// have where they are a power of two, so that the half's lowest bits
    /// pick the cluster, as a mask would; else none.
    turn: u32,
    filter: Filter,
    /// Filter insertions since it was last rebuilt from the table: evicted
    /// positions leave their bits behind. Shared, the threads count their own
    /// and add them up now and then (`count`).
    inserted: AtomicUsize,
    /// Whether the filter is read. Once a quarter of its bits are set, it
    /// turns away too few positions to pay for the wait on its word before
    /// the tables' reads, and it is dropped: every position may then be held.
    /// A rebuild that comes out sparser takes it back. Read now and then
    /// (`filtering`).
    filtering: AtomicBool,
    /// Whether threads share the table.
    shared: bool,
}

impl Table {
    /// A table of at most `megabytes`, and at least one cluster, for threads
    /// to share or not, unless its memory cannot be had.
    pub fn new(megabytes: usize, shared: bool) -> Result<Table, TryReserveError> {
        let clusters = (megabytes.saturating_mul(1 << 20) / size_of::<Cluster>()).max(1);
        // No more than `index` reaches, 2^32 (256 GB).
        let clusters = clusters.min(usize::try_from(1u64 << 32).unwrap_or(usize::MAX));
        let turn = if clusters.is_power_of_two() {
            clusters.trailing_zeros()
        } else {
            0
        };
        // Eight bits an entry, as long as the filter stays in the cache.
        let bits = (clusters * 32).next_power_of_two().clamp(1 << 20, 1 << 25);
        Ok(Table {
            clusters: large(clusters)?,
            turn,
            filter: Filter::new(bits)?,
            inserted: AtomicUsize::new(0),
            filtering: AtomicBool::new(true),
            shared,
        })
    }

    /// Starts loading the cluster a position would be in, so that a lookup
    /// soon after does not wait for memory.
    pub fn prefetch(&self, hash: u64) {
        prefetch(&self.clusters[self.index(hash)]);
    }

    /// Starts loading the filter's word for a position, so that `may_hold`
    /// soon after does not wait for memory: while the filter is read.
    pub fn prefetch_filter(&self, hash: u64) {
        self.filter.prefetch(check(hash));
    }

    /// Whether the table, or anything noted with `note`, may hold the
    /// position; `filtering` is whether the filter is read, as `filtering`
    /// last said, and so for every method that takes it.
    pub fn may_hold(&self, hash: u64, filtering: bool) -> bool {
        !filtering || self.filter.contains(check(hash))
    }

    /// Whether the filter is still read: for a thread to take again after
    /// each change (see above).
    pub fn filtering(&self) -> bool {
        self.filtering.load(Relaxed)
    }

    /// Marks a position held elsewhere, such as among the proofs. Returns
    /// whether the filter is still read.
    pub fn note(&self, hash: u64, filtering: bool) -> bool {
        self.insert(check(hash), filtering)
    }

    /// Sets a position's bit while the filter is read, and for a table of its
    /// own drops the filter once a quarter of its bits are set; shared, the
    /// search drops it once a sample says so (`filter_is_dense`). Returns
    /// whether the filter is still read.
    fn insert(&self, check: u32, filtering: bool) -> bool {
        if !filtering {
            return false;
        }
        if self.shared {
            self.filter.insert_shared(check);
        } else {
            self.filter.insert(check);
            if self.filter.set.load(Relaxed) * 4 > self.filter.bits() {
                self.filtering.store(false, Relaxed);
                return false;
            }
        }
        true
    }

    pub fn get(&self, hash: u64, filtering: bool) -> Option<Known> {
        if !self.may_hold(hash, filtering) {
            return None;
        }
        let check = check(hash);
        let (head, tail) = self.cluster(hash).slots.iter().find_map(|slot| {
            let head = slot.head.load(Relaxed);
            if self.shared {
                let tail = slot.tail.load(Relaxed);
                (head as u32 ^ mix(tail) == check).then_some((head, tail))
            } else {
                (head as u32 == check).then(|| (head, slot.tail.load(Relaxed)))
            }
        })?;
        let (first, disproof, meta) = ((head >> 32) as u32, tail as u32, (tail >> 32) as u32);
        Some(if disproof != 0 {
            if first != 0 {
                Known::Open {
                    proof: first,
                    disproof,
                }
            } else {
                Known::Mate(disproof as u16)
            }
        } else {
            Known::Refuted {
                cycle: (first != 0).then_some(first),
                from: meta & 0xff,
            }
        })
    }

    /// Stores what is known with the work behind it, over the position's old
    /// entry, an empty slot, or the entry with the least work. Returns
    /// whether the filter is still read.
    #[inline(always)]
    pub fn put(&self, hash: u64, known: Known, work: u64, filtering: bool) -> bool {
        let check = check(hash);
        let (first, disproof, from) = match known {
            Known::Open { proof, disproof } => {
                debug_assert!(proof != 0, "a win is a mate in one or kept as a proof");
                (proof, disproof.max(1), 0)
            }
            Known::Refuted { cycle, from } => (cycle.unwrap_or(0), 0, from.min(0xff)),
            // A move's bits are never zero.
            Known::Mate(m) => (0, u32::from(m), 0),
        };
        let meta = ((work.min(WORK_MAX) as u32) << 8) | from;
        let tail = u64::from(disproof) | u64::from(meta) << 32;
        let kept = if self.shared {
            check ^ mix(tail)
        } else {
            check
        };
        let head = u64::from(kept) | u64::from(first) << 32;
        let slots = &self.clusters[self.index(hash)].slots;
        let target = match slots.iter().position(|slot| self.check_of(slot) == check) {
            Some(same) => same,
            None => (0..slots.len())
                .min_by_key(|&i| {
                    let slot = &slots[i];
                    (self.check_of(slot) != 0, work_of(slot.tail.load(Relaxed)))
                })
                .unwrap_or(0),
        };
        slots[target].head.store(head, Relaxed);
        slots[target].tail.store(tail, Relaxed);
        let filtering = self.insert(check, filtering);
        if !self.shared {
            add(&self.inserted, 1, false);
        }
        filtering
    }

    /// Adds a thread's filter insertions to the count since the last
    /// rebuild, as threads that share the table do now and then.
    pub fn count(&self, insertions: usize) {
        add(&self.inserted, insertions, self.shared);
    }

    /// Shared: whether the filter is still read though a sample of it is a
    /// quarter set, so that it should be dropped (`drop_filter`).
    pub fn filter_is_dense(&self) -> bool {
        self.filtering() && self.filter.dense()
    }

    /// Shared: stops reading the filter, while the other threads wait.
    pub fn drop_filter(&self) {
        self.filtering.store(false, Relaxed);
    }

    /// The check bits of a slot's position, zero for none.
    fn check_of(&self, slot: &Slot) -> u32 {
        let head = slot.head.load(Relaxed) as u32;
        if self.shared {
            head ^ mix(slot.tail.load(Relaxed))
        } else {
            head
        }
    }

    /// The share of slots taken, from up to 65,536 clusters spread over the
    /// table.
    pub fn fill(&self) -> f64 {
        let step = (self.clusters.len() / 65_536).max(1);
        let sampled = self.clusters.iter().step_by(step);
        let (taken, slots) = sampled.fold((0, 0), |(taken, slots), cluster| {
            let held = cluster
                .slots
                .iter()
                .filter(|slot| self.check_of(slot) != 0)
                .count();
            (taken + held, slots + cluster.slots.len())
        });
        taken as f64 / slots as f64
    }

    /// Whether enough has been stored since the filter was last rebuilt to
    /// rebuild it: evicted positions leave their bits behind, and a dropped
    /// filter may come out sparse enough to read again.
    pub fn filter_is_stale(&self) -> bool {
        self.inserted.load(Relaxed) > self.clusters.len() * 8
    }

    /// Rebuilds the filter from the table and the positions held elsewhere,
    /// at most `held` of them, and reads it if it comes out sparse. A dropped
    /// filter is rebuilt only if it may: a quarter of its bits set takes
    /// about 0.29 positions a bit, and a large table holds far more. Shared,
    /// only while the other threads wait. Returns whether the filter is read.
    pub fn rebuild_filter(&self, elsewhere: impl Iterator<Item = u64>, held: usize) -> bool {
        self.inserted.store(0, Relaxed);
        let positions = self.fill() * (self.clusters.len() * 4) as f64 + held as f64;
        if !self.filtering() && positions > 0.29 * self.filter.bits() as f64 {
            return false;
        }
        self.filter.clear();
        for cluster in &self.clusters {
            for slot in &cluster.slots {
                let check = self.check_of(slot);
                if check != 0 {
                    self.filter.insert(check);
                }
            }
        }
        for hash in elsewhere {
            self.filter.insert(check(hash));
        }
        let sparse = self.filter.set.load(Relaxed) * 4 <= self.filter.bits();
        self.filtering.store(sparse, Relaxed);
        sparse
    }

    /// A position's cluster, from the low half of its hash, which its check
    /// leaves: the half's share of the clusters, by a multiply and a shift.
    fn index(&self, hash: u64) -> usize {
        let low = (hash as u32).rotate_right(self.turn);
        ((u64::from(low) * self.clusters.len() as u64) >> 32) as usize
    }

    fn cluster(&self, hash: u64) -> &Cluster {
        &self.clusters[self.index(hash)]
    }
}

/// A value on cache lines of its own: 128 bytes, as processors fetch lines in
/// pairs or have lines that long, so that threads that change it often do not
/// take the lines from those that read what would sit beside it.
#[derive(Default)]
#[repr(align(128))]
pub struct Padded<T>(pub T);

impl<T> std::ops::Deref for Padded<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

/// Adds to a count: by an atomic step where threads share it, else by a
/// plain one, which costs one thread nothing.
fn add(count: &AtomicUsize, value: usize, shared: bool) {
    if shared {
        count.fetch_add(value, Relaxed);
    } else {
        count.store(count.load(Relaxed) + value, Relaxed);
    }
}

/// Asks the processor to start loading the cache line holding `item`, without
/// waiting for it: lookups of a position's children overlap their memory
/// latencies instead of paying each in turn, which matters once the tables
/// are larger than the caches and the filter no longer turns misses away.
pub fn prefetch<T>(item: &T) {
    let address = std::ptr::from_ref(item).cast::<u8>();
    #[cfg(target_arch = "aarch64")]
    // SAFETY: a prefetch is a hint that reads nothing and cannot fault.
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{address}]",
            address = in(reg) address,
            options(nostack, preserves_flags, readonly)
        );
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: as above.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(address.cast());
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = address;
}

/// The types of the search's large tables, whose all-zero bytes are their
/// default value, which `large` sets up by writing zeros.
///
/// # Safety
///
/// All-zero bytes must be a value of the type, and its default.
pub unsafe trait Zeroed: Default {}

// SAFETY: an atomic integer has its integer's bytes, and defaults to zero.
unsafe impl Zeroed for AtomicU64 {}
// SAFETY: two atomic integers, which default to zero.
unsafe impl Zeroed for Slot {}
// SAFETY: four slots.
unsafe impl Zeroed for Cluster {}

/// `count` default values for one of the search's large tables, unless the
/// memory cannot be had, as a browser may refuse it. On Linux their pages are
/// first asked to be huge, as Stockfish asks for its hash: with small pages a
/// table of a gigabyte misses the processor's address translations on almost
/// every lookup, and Linux gives transparent huge pages by default only to
/// memory that asks for them. Where the kernel declines, the pages are the
/// usual ones.
pub fn large<T: Zeroed>(count: usize) -> Result<Vec<T>, TryReserveError> {
    let mut values = Vec::<T>::new();
    values.try_reserve_exact(count)?;
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    advise_huge_pages(values.as_mut_ptr().cast(), count * size_of::<T>());
    // SAFETY: the memory for `count` values is reserved, and zeros are the
    // default value of `T`.
    unsafe {
        values.as_mut_ptr().write_bytes(0, count);
        values.set_len(count);
    }
    Ok(values)
}

/// Asks Linux to back the whole huge pages inside these bytes with huge
/// pages, before anything is written to them.
#[cfg(all(
    target_os = "linux",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
fn advise_huge_pages(start: *mut u8, length: usize) {
    use std::ffi::{c_int, c_void};
    unsafe extern "C" {
        fn madvise(address: *mut c_void, length: usize, advice: c_int) -> c_int;
    }
    /// Linux's value on x86-64 and aarch64.
    const MADV_HUGEPAGE: c_int = 14;
    const HUGE: usize = 2 << 20;
    let first = start.addr().next_multiple_of(HUGE);
    let end = (start.addr() + length) / HUGE * HUGE;
    if end > first {
        // SAFETY: the range lies inside the allocation, and the advice
        // changes how its pages are backed, never what they hold. A kernel
        // without transparent huge pages refuses it, which changes nothing.
        unsafe {
            madvise(
                start.wrapping_add(first - start.addr()).cast(),
                end - first,
                MADV_HUGEPAGE,
            )
        };
    }
}

/// The words of the filter that `Filter::dense` reads: 4,096 bits, whose
/// share set is off the whole filter's by some 0.7 points at a quarter.
const SAMPLE: usize = 64;

/// A one-hash Bloom filter over check bits: a clear bit means a position is
/// held nowhere, which a bit in the cache tells sooner than the table.
struct Filter {
    words: Vec<AtomicU64>,
    mask: u32,
    /// The bits set.
    set: AtomicUsize,
}

impl Filter {
    fn new(bits: usize) -> Result<Filter, TryReserveError> {
        Ok(Filter {
            words: large(bits / 64)?,
            mask: (bits - 1) as u32,
            set: AtomicUsize::new(0),
        })
    }

    fn bits(&self) -> usize {
        self.words.len() * 64
    }

    /// Sets a position's bit, and counts it if it is new.
    fn insert(&self, check: u32) {
        let (word, mask) = self.bit(check);
        let old = word.load(Relaxed);
        word.store(old | mask, Relaxed);
        if old & mask == 0 {
            self.set.store(self.set.load(Relaxed) + 1, Relaxed);
        }
    }

    /// `insert` for threads that share the filter, which counts nothing: by an
    /// atomic step, so that two threads setting bits of one word keep both,
    /// and only for a clear bit, as the step takes the word's line from the
    /// other threads' caches.
    fn insert_shared(&self, check: u32) {
        let (word, mask) = self.bit(check);
        if word.load(Relaxed) & mask == 0 {
            word.fetch_or(mask, Relaxed);
        }
    }

    /// Whether a quarter of the bits are set, as far as `SAMPLE` words spread
    /// over the filter tell: positions' check bits fall anywhere alike.
    fn dense(&self) -> bool {
        let step = (self.words.len() / SAMPLE).max(1);
        let (set, words) = self
            .words
            .iter()
            .step_by(step)
            .fold((0, 0), |(set, words), word| {
                (set + word.load(Relaxed).count_ones() as usize, words + 1)
            });
        set * 4 > words * 64
    }

    /// A position's word and its bit in it.
    fn bit(&self, check: u32) -> (&AtomicU64, u64) {
        let bit = check & self.mask;
        (&self.words[(bit >> 6) as usize], 1 << (bit & 63))
    }

    fn prefetch(&self, check: u32) {
        prefetch(&self.words[((check & self.mask) >> 6) as usize]);
    }

    fn contains(&self, check: u32) -> bool {
        let bit = check & self.mask;
        self.words[(bit >> 6) as usize].load(Relaxed) & (1 << (bit & 63)) != 0
    }

    fn clear(&self) {
        for word in &self.words {
            word.store(0, Relaxed);
        }
        self.set.store(0, Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::{Known, Table};

    /// A hash whose check bits, and so whose filter bit, are `check`.
    fn hash(check: u64) -> u64 {
        check << 32 | 0x5bd1
    }

    /// The `i`th of hashes whose filter bits are spread as positions' are,
    /// each of the first 2^19 a bit of its own in a filter of 2^20, never
    /// 2^19's.
    fn spread(i: u64) -> u64 {
        hash(u64::from((i as u32).wrapping_mul(0x9e37_79b9)))
    }

    /// The filter turns away positions while it is sparse, and is dropped
    /// once a quarter of its bits are set; held positions are found either
    /// way.
    #[test]
    fn drops_a_filled_filter() {
        // One megabyte: 16,384 clusters, so the filter's least 2^20 bits.
        let table = Table::new(1, false).unwrap();
        let held = hash(3);
        let mut filtering = table.put(
            held,
            Known::Open {
                proof: 2,
                disproof: 7,
            },
            1,
            true,
        );
        let absent = hash(1 << 19);
        assert!(filtering && table.may_hold(held, filtering) && !table.may_hold(absent, filtering));
        // With the held position's, a quarter of the 2^20 bits: still read.
        for check in 4..(1 << 18) + 3 {
            filtering = table.note(hash(check), filtering);
        }
        assert!(filtering && table.filtering() && !table.may_hold(absent, filtering));
        // One bit more drops it.
        filtering = table.note(hash((1 << 18) + 3), filtering);
        assert!(!filtering && !table.filtering() && table.may_hold(absent, filtering));
        assert!(matches!(
            table.get(held, filtering),
            Some(Known::Open {
                proof: 2,
                disproof: 7
            })
        ));
        assert!(table.get(absent, filtering).is_none());
    }

    /// Shared, the filter is dense once a sample of it is a quarter set: not
    /// at a fifth, at a third. Dropped, it holds back no position.
    #[test]
    fn drops_a_shared_filter_by_a_sample() {
        let table = Table::new(1, true).unwrap();
        let held = hash(3);
        table.put(
            held,
            Known::Open {
                proof: 2,
                disproof: 7,
            },
            1,
            true,
        );
        let absent = hash(1 << 19);
        for i in 0..(1 << 20) / 5 {
            assert!(table.note(spread(i), true));
        }
        assert!(!table.filter_is_dense() && !table.may_hold(absent, true));
        for i in (1 << 20) / 5..(1 << 20) / 3 {
            table.note(spread(i), true);
        }
        assert!(table.filtering() && table.filter_is_dense());
        table.drop_filter();
        assert!(!table.filtering() && !table.filter_is_dense() && table.may_hold(absent, false));
        assert!(matches!(
            table.get(held, false),
            Some(Known::Open {
                proof: 2,
                disproof: 7
            })
        ));
    }

    /// A table of any size spreads positions over its clusters evenly, by the
    /// low half of their hash, and tells apart those in one cluster by the
    /// high half; at a power of two, the lowest bits pick the cluster.
    #[test]
    fn takes_any_size() {
        // Three megabytes: 49,152 clusters, not a power of two.
        let table = Table::new(3, false).unwrap();
        let clusters = table.clusters.len();
        assert_eq!(clusters, 49_152);
        let mut held = vec![0; clusters];
        for i in 0..clusters as u64 * 4 {
            let low = (2 * i + 1) * (1 << 31) / (clusters as u64 * 4);
            held[table.index(i << 32 | low)] += 1;
        }
        assert!(held.iter().all(|&positions| positions == 4));
        let last = |check: u64| check << 32 | 0xffff_ffff;
        assert_eq!(table.index(last(1)), clusters - 1);
        for check in 1..=4 {
            let proof = check as u32;
            table.put(last(check), Known::Open { proof, disproof: 1 }, 1, true);
        }
        for check in 1..=4 {
            let known = table.get(last(check), true);
            assert!(matches!(known, Some(Known::Open { proof, .. }) if u64::from(proof) == check));
        }
        assert!(table.get(last(5), true).is_none());
        // One megabyte: 16,384 clusters, which the lowest 14 bits pick.
        let table = Table::new(1, false).unwrap();
        let mut hash = 0x1234_5678_9abc_def0_u64;
        for _ in 0..10_000 {
            assert_eq!(table.index(hash), hash as usize & 0x3fff);
            hash = hash.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
        }
    }

    /// A dropped filter is rebuilt and read again when the positions held
    /// would set at most a quarter of its bits, and stays dropped otherwise.
    #[test]
    fn a_rebuild_takes_back_a_sparse_filter() {
        for shared in [false, true] {
            let table = Table::new(1, shared).unwrap();
            let held = hash(3);
            let mut filtering = table.put(
                held,
                Known::Open {
                    proof: 2,
                    disproof: 7,
                },
                1,
                true,
            );
            // A third of the bits: the filter is dropped, shared once found
            // dense.
            for i in 0..(1 << 20) / 3 {
                filtering = table.note(spread(i), filtering);
            }
            if table.filter_is_dense() {
                table.drop_filter();
            }
            assert!(!table.filtering());
            // Held elsewhere: too many for a quarter of the bits.
            let many = 0.3 * f64::from(1 << 20);
            assert!(!table.rebuild_filter(std::iter::empty(), many as usize));
            assert!(!table.filtering());
            // Few: the filter holds the table's position and the one elsewhere.
            let elsewhere = hash(5);
            assert!(table.rebuild_filter(std::iter::once(elsewhere), 1) && table.filtering());
            assert!(table.may_hold(held, true) && table.may_hold(elsewhere, true));
            assert!(!table.may_hold(hash(1 << 19), true) && !table.may_hold(hash(4), true));
        }
    }

    /// Shared, every kind of entry comes back as stored, a slot written over
    /// by another position's entry is a miss for the first, and an entry read
    /// half from one write and half from another is a miss too.
    #[test]
    fn shared_entries_come_back_whole() {
        let table = Table::new(1, true).unwrap();
        let (open, refuted, mate) = (hash(7), hash(8), hash(9));
        table.put(
            open,
            Known::Open {
                proof: 5,
                disproof: 11,
            },
            300,
            true,
        );
        table.put(
            refuted,
            Known::Refuted {
                cycle: Some(42),
                from: 17,
            },
            2,
            true,
        );
        table.put(mate, Known::Mate(0x1234), 1, true);
        assert!(matches!(
            table.get(open, true),
            Some(Known::Open {
                proof: 5,
                disproof: 11
            })
        ));
        assert!(matches!(
            table.get(refuted, true),
            Some(Known::Refuted {
                cycle: Some(42),
                from: 17
            })
        ));
        assert!(matches!(table.get(mate, true), Some(Known::Mate(0x1234))));
        // The open entry's slot, its first word from another write.
        let slot = table
            .cluster(open)
            .slots
            .iter()
            .find(|slot| table.check_of(slot) == 7)
            .unwrap();
        let tail = slot.tail.load(std::sync::atomic::Ordering::Relaxed);
        slot.tail
            .store(tail ^ (1 << 33), std::sync::atomic::Ordering::Relaxed);
        assert!(table.get(open, true).is_none());
        assert!(table.get(refuted, true).is_some() && table.get(mate, true).is_some());
    }

    /// Threads storing and reading at once lose no filter bit and read no
    /// entry but whole ones: every entry found is one some thread stored for
    /// that position.
    #[test]
    fn threads_share_a_table() {
        let table = Table::new(1, true).unwrap();
        let (threads, each) = (4u64, 50_000u64);
        // A few positions, so that threads meet in their slots, with check
        // bits spread as a hash's are; each entry's numbers tell its position.
        let position = |p: u64| u64::from(((2 * p + 1) as u32).wrapping_mul(0x9e37_79b9)) << 32;
        std::thread::scope(|scope| {
            for thread in 0..threads {
                let table = &table;
                scope.spawn(move || {
                    let mut x = 0x9e37_79b9_7f4a_7c15_u64 ^ thread;
                    for i in 0..each * 4 {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        let p = x % 4096;
                        let proof = (p as u32 + 1) * 4 + (i % 4) as u32;
                        let entry = Known::Open {
                            proof,
                            disproof: proof,
                        };
                        table.put(position(p) | x >> 52, entry, i, true);
                        if i % 4 == 0 {
                            table.note(hash(4096 + thread * each + i / 4), true);
                        }
                        let p = (x >> 20) % 4096;
                        if let Some(known) = table.get(position(p) | x >> 52, true) {
                            let Known::Open { proof, disproof } = known else {
                                panic!("only open entries were stored");
                            };
                            assert_eq!((proof, u64::from(proof / 4 - 1)), (disproof, p));
                        }
                    }
                });
            }
        });
        // Under a quarter of the filter's bits: it is still read, and holds
        // every noted position.
        assert!(table.filtering() && !table.filter_is_dense());
        for position in 4096..4096 + threads * each {
            assert!(table.may_hold(hash(position), true));
        }
    }
}
