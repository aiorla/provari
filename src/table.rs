//! The search's bounded memory: a fixed-size table of proof and disproof
//! numbers, in clusters of four entries that share a cache line, behind a
//! filter of the positions it may hold while the filter is sparse. A new
//! entry replaces the one whose search took the least work, the cheapest to
//! redo.

use std::collections::TryReserveError;

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

/// One entry: a proof and a disproof number, for a refutation (disproof zero)
/// the repeated position's check bits, or for a mate in one (proof zero) its
/// move's bits; then the work and the counter.
#[derive(Clone, Copy, Default)]
struct Slot {
    check: u32,
    first: u32,
    disproof: u32,
    /// The work behind the entry, saturating, above the refutation's counter.
    meta: u32,
}

const WORK_MAX: u64 = (1 << 24) - 1;

impl Slot {
    fn work(&self) -> u32 {
        self.meta >> 8
    }
}

#[derive(Clone, Copy, Default)]
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
    /// positions leave their bits behind.
    inserted: usize,
    /// Whether the filter is read. Once a quarter of its bits are set, it
    /// turns away too few positions to pay for the wait on its word before
    /// the tables' reads, and it is dropped: every position may then be held.
    /// A rebuild that comes out sparser takes it back.
    filtering: bool,
}

impl Table {
    /// A table of at most `megabytes`, and at least one cluster, unless its
    /// memory cannot be had.
    pub fn new(megabytes: usize) -> Result<Table, TryReserveError> {
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
            inserted: 0,
            filtering: true,
        })
    }

    /// Starts loading the cluster a position would be in, so that a lookup
    /// soon after does not wait for memory.
    pub fn prefetch(&self, hash: u64) {
        prefetch(&self.clusters[self.index(hash)]);
    }

    /// Starts loading the filter's word for a position, so that `may_hold`
    /// soon after does not wait for memory.
    pub fn prefetch_filter(&self, hash: u64) {
        if self.filtering {
            self.filter.prefetch(check(hash));
        }
    }

    /// Whether the table, or anything noted with `note`, may hold the position.
    pub fn may_hold(&self, hash: u64) -> bool {
        !self.filtering || self.filter.contains(check(hash))
    }

    /// Whether the filter is still read.
    pub fn filtering(&self) -> bool {
        self.filtering
    }

    /// Marks a position held elsewhere, such as among the proofs.
    pub fn note(&mut self, hash: u64) {
        self.insert(check(hash));
    }

    /// Sets a position's bit while the filter is read, and drops the filter
    /// once a quarter of its bits are set.
    fn insert(&mut self, check: u32) {
        if !self.filtering {
            return;
        }
        self.filter.insert(check);
        if self.filter.set * 4 > self.filter.bits() {
            self.filtering = false;
        }
    }

    pub fn get(&self, hash: u64) -> Option<Known> {
        if !self.may_hold(hash) {
            return None;
        }
        let check = check(hash);
        let slot = self.cluster(hash).slots.iter().find(|s| s.check == check)?;
        Some(if slot.disproof != 0 {
            if slot.first != 0 {
                Known::Open {
                    proof: slot.first,
                    disproof: slot.disproof,
                }
            } else {
                Known::Mate(slot.disproof as u16)
            }
        } else {
            Known::Refuted {
                cycle: (slot.first != 0).then_some(slot.first),
                from: slot.meta & 0xff,
            }
        })
    }

    /// Stores what is known with the work behind it, over the position's old
    /// entry, an empty slot, or the entry with the least work.
    #[inline(always)]
    pub fn put(&mut self, hash: u64, known: Known, work: u64) {
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
        let slot = Slot {
            check,
            first,
            disproof,
            meta: ((work.min(WORK_MAX) as u32) << 8) | from,
        };
        let index = self.index(hash);
        let slots = &mut self.clusters[index].slots;
        let target = match slots.iter().position(|s| s.check == check) {
            Some(same) => same,
            None => (0..slots.len())
                .min_by_key(|&i| (slots[i].check != 0, slots[i].work()))
                .unwrap_or(0),
        };
        slots[target] = slot;
        self.insert(check);
        self.inserted += 1;
    }

    /// The share of slots taken, from up to 65,536 clusters spread over the
    /// table.
    pub fn fill(&self) -> f64 {
        let step = (self.clusters.len() / 65_536).max(1);
        let sampled = self.clusters.iter().step_by(step);
        let (taken, slots) = sampled.fold((0, 0), |(taken, slots), cluster| {
            let held = cluster.slots.iter().filter(|slot| slot.check != 0).count();
            (taken + held, slots + cluster.slots.len())
        });
        taken as f64 / slots as f64
    }

    /// Whether enough has been stored since the filter was last rebuilt to
    /// rebuild it: evicted positions leave their bits behind, and a dropped
    /// filter may come out sparse enough to read again.
    pub fn filter_is_stale(&self) -> bool {
        self.inserted > self.clusters.len() * 8
    }

    /// Rebuilds the filter from the table and the positions held elsewhere,
    /// at most `held` of them, and reads it if it comes out sparse. A dropped
    /// filter is rebuilt only if it may: a quarter of its bits set takes
    /// about 0.29 positions a bit, and a large table holds far more.
    pub fn rebuild_filter(&mut self, elsewhere: impl Iterator<Item = u64>, held: usize) {
        self.inserted = 0;
        let positions = self.fill() * (self.clusters.len() * 4) as f64 + held as f64;
        if !self.filtering && positions > 0.29 * self.filter.bits() as f64 {
            return;
        }
        self.filter.clear();
        for cluster in &self.clusters {
            for slot in &cluster.slots {
                if slot.check != 0 {
                    self.filter.insert(slot.check);
                }
            }
        }
        for hash in elsewhere {
            self.filter.insert(check(hash));
        }
        self.filtering = self.filter.set * 4 <= self.filter.bits();
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

/// `count` default values for one of the search's large tables, unless the
/// memory cannot be had, as a browser may refuse it. On Linux their pages are
/// first asked to be huge, as Stockfish asks for its hash: with small pages a
/// table of a gigabyte misses the processor's address translations on almost
/// every lookup, and Linux gives transparent huge pages by default only to
/// memory that asks for them. Where the kernel declines, the pages are the
/// usual ones.
pub fn large<T: Copy + Default>(count: usize) -> Result<Vec<T>, TryReserveError> {
    let mut values = Vec::<T>::new();
    values.try_reserve_exact(count)?;
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    advise_huge_pages(values.as_mut_ptr().cast(), count * size_of::<T>());
    values.resize(count, T::default());
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

/// A one-hash Bloom filter over check bits: a clear bit means a position is
/// held nowhere, which a bit in the cache tells sooner than the table.
struct Filter {
    words: Vec<u64>,
    mask: u32,
    /// The bits set.
    set: usize,
}

impl Filter {
    fn new(bits: usize) -> Result<Filter, TryReserveError> {
        Ok(Filter {
            words: large(bits / 64)?,
            mask: (bits - 1) as u32,
            set: 0,
        })
    }

    fn bits(&self) -> usize {
        self.words.len() * 64
    }

    fn insert(&mut self, check: u32) {
        let bit = check & self.mask;
        let word = &mut self.words[(bit >> 6) as usize];
        let mask = 1 << (bit & 63);
        self.set += usize::from(*word & mask == 0);
        *word |= mask;
    }

    fn prefetch(&self, check: u32) {
        prefetch(&self.words[((check & self.mask) >> 6) as usize]);
    }

    fn contains(&self, check: u32) -> bool {
        let bit = check & self.mask;
        self.words[(bit >> 6) as usize] & (1 << (bit & 63)) != 0
    }

    fn clear(&mut self) {
        self.words.fill(0);
        self.set = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::{Known, Table};

    /// A hash whose check bits, and so whose filter bit, are `check`.
    fn hash(check: u64) -> u64 {
        check << 32 | 0x5bd1
    }

    /// The filter turns away positions while it is sparse, and is dropped
    /// once a quarter of its bits are set; held positions are found either
    /// way.
    #[test]
    fn drops_a_filled_filter() {
        // One megabyte: 16,384 clusters, so the filter's least 2^20 bits.
        let mut table = Table::new(1).unwrap();
        let held = hash(3);
        table.put(
            held,
            Known::Open {
                proof: 2,
                disproof: 7,
            },
            1,
        );
        let absent = hash(1 << 19);
        assert!(table.filtering() && table.may_hold(held) && !table.may_hold(absent));
        // With the held position's, a quarter of the 2^20 bits: still read.
        for check in 4..(1 << 18) + 3 {
            table.note(hash(check));
        }
        assert!(table.filtering() && !table.may_hold(absent));
        // One bit more drops it.
        table.note(hash((1 << 18) + 3));
        assert!(!table.filtering() && table.may_hold(absent));
        assert!(matches!(
            table.get(held),
            Some(Known::Open {
                proof: 2,
                disproof: 7
            })
        ));
        assert!(table.get(absent).is_none());
    }

    /// A table of any size spreads positions over its clusters evenly, by the
    /// low half of their hash, and tells apart those in one cluster by the
    /// high half; at a power of two, the lowest bits pick the cluster.
    #[test]
    fn takes_any_size() {
        // Three megabytes: 49,152 clusters, not a power of two.
        let mut table = Table::new(3).unwrap();
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
            table.put(last(check), Known::Open { proof, disproof: 1 }, 1);
        }
        for check in 1..=4 {
            let known = table.get(last(check));
            assert!(matches!(known, Some(Known::Open { proof, .. }) if u64::from(proof) == check));
        }
        assert!(table.get(last(5)).is_none());
        // One megabyte: 16,384 clusters, which the lowest 14 bits pick.
        let table = Table::new(1).unwrap();
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
        let mut table = Table::new(1).unwrap();
        let held = hash(3);
        table.put(
            held,
            Known::Open {
                proof: 2,
                disproof: 7,
            },
            1,
        );
        for check in 4..(1 << 18) + 4 {
            table.note(hash(check));
        }
        assert!(!table.filtering());
        // Held elsewhere: too many for a quarter of the bits.
        let many = 0.3 * f64::from(1 << 20);
        table.rebuild_filter(std::iter::empty(), many as usize);
        assert!(!table.filtering());
        // Few: the filter holds the table's position and the one elsewhere.
        let elsewhere = hash(5);
        table.rebuild_filter(std::iter::once(elsewhere), 1);
        assert!(table.filtering() && table.may_hold(held) && table.may_hold(elsewhere));
        assert!(!table.may_hold(hash(1 << 19)) && !table.may_hold(hash(4)));
    }
}
