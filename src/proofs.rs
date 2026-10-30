//! Proven positions in a fixed amount of memory. An entry keeps a position's
//! whole key, the winning move where the attacker is to move, the proof's
//! length in attacker moves and the highest 50-move counter it holds at, in 12
//! bytes, in an ordered open-addressing table (Amble and Knuth 1974): an
//! entry sits at or after its key's place, with every slot between them
//! taken, and the keys of every run of taken slots rise along it. A lookup
//! must see each of a position's proofs, so it reads from the key's place to
//! the first key above its own, where an unordered table reads to the end of
//! the run: at nine tenths full, some 5 slots against 50. A key's place rises
//! with the key, so a run in key order keeps every entry at or after its
//! place: an insertion moves the rest of its run up a slot, and a deletion
//! moves it down to the first proof at its own place. A position keeps its
//! shortest proof and any longer ones that hold at higher counters.
//!
//! Proofs are not dropped one by one: no line of a proof can repeat a position
//! because lengths shrink along it, and a proof found again after being
//! dropped could be longer than its parent assumed. So the table takes proofs
//! until it is nine tenths full, then refuses them; its lookups slow down as
//! it fills, which still beats stopping. Mates in one are marked, so that the
//! search can drop them all when the table first fills (`drop_mates`) and work
//! each out again where it needs one.
//!
//! Threads may share the table (`shared`). Its slots then fall into stretches
//! of `STRETCH`, each with a version: a thread that changes slots locks every
//! stretch they are in, in order, making their versions odd, and unlocks them
//! with the next even ones. A lookup reads the versions of the stretches it
//! walks before and after its walk, and walks again if one changed, so that
//! it never sees half a move of a run. Dropping the mates in one, which moves
//! every proof, waits until no other thread is in the table.

use std::collections::TryReserveError;
use std::sync::atomic::{
    AtomicU32, AtomicUsize,
    Ordering::{Acquire, Relaxed, Release},
    fence,
};

use crate::board::Move;
use crate::table::{Padded, Zeroed};

/// A proven position's winning move where the attacker is to move, its length
/// in attacker moves, and the highest 50-move counter it holds at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Proof {
    pub m: Option<Move>,
    pub length: u32,
    pub limit: u32,
}

/// A proof as the table keeps it: the key, then in 32 bits the winning move in
/// 15 (0 for none, as no move goes from a square to itself), the length in 10
/// and the limit in 7: a counter up to `LAST`, `MATE` for a mate in one, which
/// holds up to `LAST`, or `ANY` for any.
#[derive(Clone, Copy, Default)]
struct Entry {
    key: u64,
    rest: u32,
}

/// A slot, which holds an entry in 12 bytes, word by word, so that threads can
/// share the table.
#[derive(Default)]
#[repr(C)]
struct Slot {
    key: [AtomicU32; 2],
    rest: AtomicU32,
}

const _: () = assert!(size_of::<Slot>() == 12);

const MOVE_BITS: u32 = 15;
const LENGTH_BITS: u32 = 10;
const LAST: u32 = 99;
const MATE: u32 = 100;
const ANY: u32 = 127;

/// The key of a free slot; positions with that key are kept under another.
const FREE: u64 = 0;

/// The slots of a stretch, which shares a version: 24 KB, so that the versions
/// of the largest tables still fit in a core's cache.
const STRETCH: usize = 1 << 11;

impl Entry {
    fn new(key: u64, proof: Proof, mate: bool) -> Entry {
        debug_assert!(proof.limit <= LAST || proof.limit == u32::MAX);
        debug_assert!(!mate || (proof.length == 1 && proof.limit == LAST));
        debug_assert_ne!(proof.m, Some(Move::default()));
        assert!(
            proof.length < 1 << LENGTH_BITS,
            "a proof fits in 1,023 moves"
        );
        let m = u32::from(proof.m.map_or(0, Move::bits));
        debug_assert!(m < 1 << MOVE_BITS);
        let limit = if mate { MATE } else { proof.limit.min(ANY) };
        Entry {
            key,
            rest: m | proof.length << MOVE_BITS | limit << (MOVE_BITS + LENGTH_BITS),
        }
    }

    fn length(&self) -> u32 {
        (self.rest >> MOVE_BITS) & ((1 << LENGTH_BITS) - 1)
    }

    /// The highest counter the proof holds at, `ANY` above every counter.
    fn limit(&self) -> u32 {
        match self.kept_limit() {
            MATE => LAST,
            limit => limit,
        }
    }

    /// Whether the proof is a mate in one, which `drop_mates` drops.
    fn mate(&self) -> bool {
        self.kept_limit() == MATE
    }

    /// The limit's bits, `MATE` for a mate in one.
    fn kept_limit(&self) -> u32 {
        self.rest >> (MOVE_BITS + LENGTH_BITS)
    }

    fn proof(&self) -> Proof {
        let m = (self.rest & ((1 << MOVE_BITS) - 1)) as u16;
        Proof {
            m: (m != 0).then(|| Move::from_bits(m)),
            length: self.length(),
            limit: match self.limit() {
                ANY => u32::MAX,
                limit => limit,
            },
        }
    }
}

impl Slot {
    fn key(&self) -> u64 {
        u64::from(self.key[0].load(Relaxed)) | u64::from(self.key[1].load(Relaxed)) << 32
    }

    fn load(&self) -> Entry {
        Entry {
            key: self.key(),
            rest: self.rest.load(Relaxed),
        }
    }

    fn store(&self, entry: Entry) {
        self.key[0].store(entry.key as u32, Relaxed);
        self.key[1].store((entry.key >> 32) as u32, Relaxed);
        self.rest.store(entry.rest, Relaxed);
    }

    /// Frees the slot, leaving the rest of its entry, which nothing reads.
    fn free(&self) {
        self.key[0].store(0, Relaxed);
        self.key[1].store(0, Relaxed);
    }
}

// SAFETY: three atomic integers, which default to zero.
unsafe impl Zeroed for Slot {}
// SAFETY: an atomic integer has its integer's bytes, and defaults to zero.
unsafe impl Zeroed for AtomicU32 {}

pub struct Proofs {
    /// The slots: the places keys start from, then room for the runs that go
    /// on past the last place, then a slot that stays free to end them.
    slots: Vec<Slot>,
    /// How many places there are.
    places: usize,
    /// Proofs held, apart from the rest, as threads that share the table
    /// change it with every proof; and the most there may be.
    used: Padded<AtomicUsize>,
    most: usize,
    /// The stretches' versions when threads share the table, else none.
    versions: Vec<AtomicU32>,
}

impl Proofs {
    /// A table of `megabytes`, and at least a few entries, with the slots
    /// past its last place (at most 768 KB more), for threads to share or
    /// not, unless its memory cannot be had.
    pub fn new(megabytes: usize, shared: bool) -> Result<Proofs, TryReserveError> {
        let places = (megabytes.saturating_mul(1 << 20) / size_of::<Slot>()).max(4);
        Proofs::with_places(places, shared)
    }

    fn with_places(places: usize, shared: bool) -> Result<Proofs, TryReserveError> {
        let most = places - (places / 10).max(1);
        let slots = places + beyond(most);
        Ok(Proofs {
            slots: crate::table::large(slots)?,
            places,
            used: Padded(AtomicUsize::new(0)),
            most,
            versions: if shared {
                crate::table::large(slots.div_ceil(STRETCH))?
            } else {
                Vec::new()
            },
        })
    }

    /// A table of `places` places, for tests.
    #[cfg(test)]
    pub fn sized(places: usize, shared: bool) -> Proofs {
        Proofs::with_places(places, shared).unwrap()
    }

    fn shared(&self) -> bool {
        !self.versions.is_empty()
    }

    /// The shortest proof of a position that holds at this counter.
    pub fn get(&self, hash: u64, counter: u32) -> Option<Proof> {
        let key = key(hash);
        let start = self.start(key);
        if self.shared() {
            return self.read(start, key, counter);
        }
        let mut best: Option<Proof> = None;
        for slot in &self.slots[start..] {
            let held = slot.key();
            if held == FREE || held > key {
                break;
            }
            if held == key {
                let proof = Entry {
                    key,
                    rest: slot.rest.load(Relaxed),
                }
                .proof();
                if proof.limit >= counter && best.is_none_or(|known| proof.length < known.length) {
                    best = Some(proof);
                }
            }
        }
        best
    }

    /// `get` while threads share the table: the walk from the key's place,
    /// again until no stretch it read changed meanwhile.
    fn read(&self, start: usize, key: u64, counter: u32) -> Option<Proof> {
        let mut walks = 0u32;
        'walk: loop {
            walks += 1;
            wait(walks);
            // The stretches read and their versions; a run seldom crosses one
            // stretch's end, and one that crosses three is read locked.
            let mut read = [(0, 0); 4];
            let mut stretches = 0;
            let mut best: Option<Proof> = None;
            for (index, slot) in self.slots.iter().enumerate().skip(start) {
                let stretch = index / STRETCH;
                if stretches == 0 || read[stretches - 1].0 != stretch {
                    if stretches == read.len() {
                        return self.read_locked(start, key, counter);
                    }
                    let version = self.versions[stretch].load(Acquire);
                    if version & 1 != 0 {
                        continue 'walk;
                    }
                    read[stretches] = (stretch, version);
                    stretches += 1;
                }
                let entry = slot.load();
                if entry.key == FREE || entry.key > key {
                    break;
                }
                if entry.key == key {
                    let proof = entry.proof();
                    if proof.limit >= counter
                        && best.is_none_or(|known| proof.length < known.length)
                    {
                        best = Some(proof);
                    }
                }
            }
            // The walk's reads before the versions' second reading.
            fence(Acquire);
            let unchanged = read[..stretches]
                .iter()
                .all(|&(stretch, version)| self.versions[stretch].load(Relaxed) == version);
            if unchanged {
                return best;
            }
        }
    }

    /// `get` with the stretches it walks locked.
    #[cold]
    fn read_locked(&self, start: usize, key: u64, counter: u32) -> Option<Proof> {
        let mut locks = Locks::new(&self.versions, start);
        let mut best: Option<Proof> = None;
        let mut index = start;
        loop {
            let entry = self.slots[index].load();
            if entry.key == FREE || entry.key > key {
                return best;
            }
            if entry.key == key {
                let proof = entry.proof();
                if proof.limit >= counter && best.is_none_or(|known| proof.length < known.length) {
                    best = Some(proof);
                }
            }
            index += 1;
            locks.reach(index);
        }
    }

    /// Adds a proof unless a known one is as short and holds as far, and drops
    /// the known ones it beats on both. Returns false if the table is too full
    /// to take it.
    pub fn add(&self, hash: u64, proof: Proof) -> bool {
        self.insert(Entry::new(key(hash), proof, false))
    }

    /// Adds a mate in one, as `add` does, marked for `drop_mates`.
    pub fn add_mate(&self, hash: u64, proof: Proof) -> bool {
        self.insert(Entry::new(key(hash), proof, true))
    }

    fn insert(&self, entry: Entry) -> bool {
        let key = entry.key;
        // Past the smaller keys of the run from the key's place to the
        // position's own proofs: a known one as good ends the search, and the
        // ones the new proof beats give up their slots, the first to it.
        let mut index = self.start(key);
        let mut locks = Locks::new(&self.versions, index);
        while let held = self.slots[index].key()
            && held != FREE
            && held < key
        {
            index += 1;
            locks.reach(index);
        }
        let mut slot = None;
        while self.slots[index].key() == key {
            let known = self.slots[index].load();
            if known.length() <= entry.length() && known.limit() >= entry.limit() {
                return true;
            }
            if known.length() >= entry.length() && known.limit() <= entry.limit() {
                if slot.is_some() {
                    // The next proof moves into its slot.
                    self.delete(index, &mut locks);
                    continue;
                }
                slot = Some(index);
            }
            index += 1;
            locks.reach(index);
        }
        if let Some(slot) = slot {
            self.slots[slot].store(entry);
            return true;
        }
        // Else in its turn, the larger keys after it moving up a slot into
        // the free one that ends the run, while the table has room: a tenth
        // of it stays free, so runs end, and the last slot stays free.
        let mut end = index;
        while self.slots[end].key() != FREE {
            end += 1;
            locks.reach(end);
        }
        if end + 1 == self.slots.len() || !self.take() {
            return false;
        }
        for index in (index..end).rev() {
            self.slots[index + 1].store(self.slots[index].load());
        }
        self.slots[index].store(entry);
        true
    }

    /// Counts a new proof in, unless the table holds the most it takes:
    /// shared, by one atomic step, whose count before says whether it may.
    fn take(&self) -> bool {
        if self.shared() {
            let room = self.used.fetch_add(1, Relaxed) < self.most;
            if !room {
                self.used.fetch_sub(1, Relaxed);
            }
            return room;
        }
        let used = self.used.load(Relaxed);
        if used >= self.most {
            return false;
        }
        self.used.store(used + 1, Relaxed);
        true
    }

    /// Drops a position's proofs.
    pub fn remove(&self, hash: u64) {
        let key = key(hash);
        let mut index = self.start(key);
        let mut locks = Locks::new(&self.versions, index);
        while let held = self.slots[index].key()
            && held != FREE
            && held < key
        {
            index += 1;
            locks.reach(index);
        }
        while self.slots[index].key() == key {
            self.delete(index, &mut locks);
        }
    }

    /// Empties a taken slot, the rest of its run moving down a slot up to
    /// the first proof at its own place: the proofs after that one are at
    /// or past it, and so need no slot before it.
    fn delete(&self, index: usize, locks: &mut Locks) {
        let mut end = index + 1;
        locks.reach(end);
        while let held = self.slots[end].key()
            && held != FREE
            && self.start(held) < end
        {
            end += 1;
            locks.reach(end);
        }
        for index in index + 1..end {
            self.slots[index - 1].store(self.slots[index].load());
        }
        self.slots[end - 1].store(Entry::default());
        if self.shared() {
            self.used.fetch_sub(1, Relaxed);
        } else {
            self.used.store(self.used.load(Relaxed) - 1, Relaxed);
        }
    }

    /// Drops every mate in one and packs the rest down in place: the slots'
    /// keys rise from the first to the last taken one, so each proof in turn
    /// moves to its key's place or the slot after the proof before it,
    /// whichever is later, at worst its own, and every run stays whole and in
    /// order. Returns how many mates it dropped. Shared, only while no other
    /// thread is in the table.
    pub fn drop_mates(&self) -> usize {
        let mut dropped = 0;
        let mut next = 0;
        for index in 0..self.slots.len() {
            let entry = self.slots[index].load();
            if entry.key == FREE {
                continue;
            }
            self.slots[index].free();
            if entry.mate() {
                dropped += 1;
                continue;
            }
            let slot = self.start(entry.key).max(next);
            self.slots[slot].store(entry);
            next = slot + 1;
        }
        self.used.store(self.used.load(Relaxed) - dropped, Relaxed);
        dropped
    }

    /// Starts loading the first slot of a position's run (see
    /// `table::prefetch`), and the next one, whose line holds the end of the
    /// first where it crosses into another, and its stretch's version.
    pub fn prefetch(&self, hash: u64) {
        let start = self.start(key(hash));
        crate::table::prefetch(&self.slots[start]);
        crate::table::prefetch(&self.slots[start + 1]);
        if self.shared() {
            crate::table::prefetch(&self.versions[start / STRETCH]);
        }
    }

    /// The proofs held: at least the positions.
    pub fn taken(&self) -> usize {
        self.used.load(Relaxed)
    }

    /// How full the table is, as a share of the most it takes.
    pub fn fill(&self) -> f64 {
        self.taken() as f64 / self.most as f64
    }

    /// The most slots it takes.
    pub fn room(&self) -> usize {
        self.most
    }

    /// The keys of every position held, for rebuilding a filter. Shared,
    /// only while no other thread is in the table.
    pub fn keys(&self) -> impl Iterator<Item = u64> + '_ {
        self.slots.iter().map(Slot::key).filter(|&key| key != FREE)
    }

    /// A key's place: any number of places, by multiplying, which keeps the
    /// keys' order.
    fn start(&self, key: u64) -> usize {
        ((u128::from(key) * self.places as u128) >> 64) as usize
    }
}

/// The stretches a change to the table has locked, from the one of its first
/// slot on; dropped, it unlocks them. Every change locks stretches in the
/// order of their slots, so that no two wait on each other. With one thread
/// there are no versions, and it locks nothing.
struct Locks<'a> {
    versions: &'a [AtomicU32],
    first: usize,
    /// The stretch after the last one locked.
    end: usize,
}

impl<'a> Locks<'a> {
    fn new(versions: &'a [AtomicU32], slot: usize) -> Locks<'a> {
        let first = slot / STRETCH;
        let mut locks = Locks {
            versions,
            first,
            end: first,
        };
        locks.reach(slot);
        locks
    }

    /// Locks every stretch up to the slot's, before the slot is read.
    #[inline(always)]
    fn reach(&mut self, slot: usize) {
        if self.versions.is_empty() {
            return;
        }
        while self.end <= slot / STRETCH {
            lock(&self.versions[self.end]);
            self.end += 1;
        }
    }
}

impl Drop for Locks<'_> {
    fn drop(&mut self) {
        if self.versions.is_empty() {
            return;
        }
        for version in &self.versions[self.first..self.end] {
            // The changes before the version a lookup reads after its walk.
            version.store(version.load(Relaxed).wrapping_add(1), Release);
        }
    }
}

/// Takes a stretch's lock: an odd version, set before any change in it.
#[cold]
fn lock(version: &AtomicU32) {
    let mut tries = 0u32;
    loop {
        let even = version.load(Relaxed) & !1;
        if version
            .compare_exchange_weak(even, even.wrapping_add(1), Acquire, Relaxed)
            .is_ok()
        {
            break;
        }
        tries += 1;
        wait(tries);
    }
    // The odd version before the changes: a lookup that reads a change reads
    // it too, after its walk.
    fence(Release);
}

/// Waits before a thread's next try at a stretch another is changing: a
/// moment, and after many tries its turn, in case the other thread waits for
/// a core.
fn wait(tries: u32) {
    if tries.is_multiple_of(64) {
        std::thread::yield_now();
    } else if tries > 1 {
        std::hint::spin_loop();
    }
}

/// The slots past the last place for the runs that go on from there, and
/// the free one that ends them: as many as the table takes, up to 65,536,
/// far longer than any run gets at nine tenths full. A run that would reach
/// the last slot all the same is refused, as a full table is.
fn beyond(most: usize) -> usize {
    most.min(1 << 16) + 1
}

/// A position's key: its hash, apart from the free slot's.
fn key(hash: u64) -> u64 {
    match hash {
        FREE => 1,
        hash => hash,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::Ordering::Relaxed;

    use super::{FREE, Proof, Proofs, STRETCH};
    use crate::board::Move;

    fn proof(length: u32, limit: u32) -> Proof {
        Proof {
            m: None,
            length,
            limit,
        }
    }

    /// Every taken slot's key at or after its place, with every slot between
    /// them taken, the keys rising along the table, the last slot free, and
    /// the slots taken counted; shared, every stretch unlocked.
    fn assert_ordered(proofs: &Proofs) {
        let mut last = FREE;
        let mut run = 0;
        let mut taken = 0;
        for (index, slot) in proofs.slots.iter().enumerate() {
            let key = slot.key();
            if key == FREE {
                run = index + 1;
                continue;
            }
            taken += 1;
            let place = proofs.start(key);
            assert!(
                run <= place && place <= index,
                "{index}: place {place}, run from {run}"
            );
            assert!(key >= last, "{index}: {key:x} after {last:x}");
            last = key;
        }
        assert_eq!(proofs.slots.last().map(|slot| slot.key()), Some(FREE));
        assert_eq!(proofs.taken(), taken);
        assert!(
            proofs
                .versions
                .iter()
                .all(|version| version.load(Relaxed) % 2 == 0)
        );
    }

    /// The shortest proof that holds at the counter, among a position's
    /// proofs at different counters.
    #[test]
    fn keeps_the_shortest_proof_for_each_counter() {
        for shared in [false, true] {
            let proofs = Proofs::new(1, shared).unwrap();
            assert!(proofs.add(7, proof(5, 40)));
            assert!(proofs.add(7, proof(9, 80)));
            assert_eq!(proofs.get(7, 30).map(|p| p.length), Some(5));
            assert_eq!(proofs.get(7, 60).map(|p| p.length), Some(9));
            assert!(proofs.get(7, 90).is_none());
            // A proof as short that holds further replaces both.
            assert!(proofs.add(7, proof(5, 90)));
            assert_eq!(proofs.get(7, 85).map(|p| p.length), Some(5));
            assert_eq!(proofs.keys().filter(|&key| key == 7).count(), 1);
            // A longer one that holds less far adds nothing.
            assert!(proofs.add(7, proof(6, 50)));
            assert_eq!(proofs.keys().filter(|&key| key == 7).count(), 1);
            assert!(proofs.get(8, 0).is_none());
            assert_eq!(proofs.get(0, 1000).map(|p| p.limit), None);
            assert!(proofs.add(0, proof(0, u32::MAX)));
            assert_eq!(proofs.get(0, 1000).map(|p| p.limit), Some(u32::MAX));
            assert!(proofs.add(u64::MAX, proof(3, 99)));
            assert_eq!(proofs.get(u64::MAX, 0).map(|p| p.length), Some(3));
            assert_ordered(&proofs);
        }
    }

    /// A proof comes back as it went in: every bit a move uses, lengths up to
    /// 1,023 and limits up to 99 or any.
    #[test]
    fn keeps_proofs_whole() {
        for shared in [false, true] {
            let proofs = Proofs::new(1, shared).unwrap();
            let whole = Proof {
                m: Some(Move::from_bits(0x7fff)),
                length: 1023,
                limit: 0,
            };
            assert!(proofs.add(3, whole));
            assert_eq!(proofs.get(3, 0), Some(whole));
            let any = Proof {
                m: Some(Move::from_bits(1)),
                length: 300,
                limit: u32::MAX,
            };
            assert!(proofs.add(5, any));
            assert_eq!(proofs.get(5, 1000), Some(any));
            assert!(proofs.add(5, proof(299, 99)));
            assert_eq!(proofs.get(5, 99), Some(proof(299, 99)));
        }
    }

    /// Positions whose keys start at the same place all keep their proofs,
    /// and the table refuses proofs past nine tenths full.
    #[test]
    fn fills_to_nine_tenths() {
        for shared in [false, true] {
            let proofs = Proofs::new(0, shared).unwrap();
            let size = proofs.places;
            // Keys spread over the places, and some that share them.
            let keys: Vec<u64> = (1..=size as u64)
                .map(|k| k.wrapping_mul(1 << 61) | k)
                .collect();
            let fit = size - (size / 10).max(1);
            for &key in &keys[..fit] {
                assert!(proofs.add(key, proof(2, 99)));
            }
            assert!(!proofs.add(keys[fit], proof(2, 99)));
            for &key in &keys[..fit] {
                assert!(proofs.get(key, 0).is_some(), "{key}");
            }
            assert!(proofs.get(keys[fit], 0).is_none());
            assert_ordered(&proofs);
        }
    }

    /// Proofs that arrive in any order land in key order and are each found
    /// again, the larger keys after each moving up a slot for it, also across
    /// the stretches of a shared table.
    #[test]
    fn keeps_each_run_in_key_order() {
        for (places, shared) in [(1024, false), (1024, true), (3 * STRETCH, true)] {
            let proofs = Proofs::sized(places, shared);
            let mut key = 0x9e37_79b9_7f4a_7c15_u64;
            let mut added = Vec::new();
            while added.len() < proofs.room() {
                key = key
                    .wrapping_mul(0x5851_f42d_4c95_7f2d)
                    .wrapping_add(1442695040888963407);
                // A third of them close to the last, to share its places.
                let kept = if added.len() % 3 == 0 {
                    key | 0xfff0 << 48
                } else {
                    key
                };
                assert!(proofs.add(kept, proof(added.len() as u32 % 1000 + 1, 99)));
                added.push(kept);
            }
            assert!(!proofs.add(key ^ 1, proof(1, 99)));
            assert_ordered(&proofs);
            for (i, &kept) in added.iter().enumerate() {
                let found = proofs.get(kept, 0);
                assert_eq!(found.map(|p| p.length), Some(i as u32 % 1000 + 1));
                if shared {
                    // As a lookup that crosses too many stretches reads it.
                    let key = super::key(kept);
                    assert_eq!(proofs.read_locked(proofs.start(key), key, 0), found);
                }
            }
        }
    }

    /// Dropping the mates in one leaves every other proof where a lookup
    /// finds it, in runs that share slots and go on past the last place, and
    /// frees their slots.
    #[test]
    fn drops_mates_and_keeps_the_rest() {
        for shared in [false, true] {
            let proofs = Proofs::sized(64, shared);
            // 64 places: a key's top six bits are its place. Places near the
            // end, where the runs go on past it, and at the start.
            let places = [62, 63, 63, 61, 0, 63, 1, 62, 0, 2, 63, 60, 1, 3, 62, 63];
            let mut kept = Vec::new();
            let mut mates = Vec::new();
            for (i, &place) in places.iter().enumerate() {
                let key = (place << 58) | (i as u64 + 1);
                if i % 3 == 1 {
                    let mate = Proof {
                        m: Some(Move::from_bits(0x0123)),
                        length: 1,
                        limit: 99,
                    };
                    assert!(proofs.add_mate(key, mate));
                    assert_eq!(proofs.get(key, 99), Some(mate));
                    mates.push(key);
                } else {
                    assert!(proofs.add(key, proof(i as u32 + 2, 99)));
                    kept.push(key);
                }
            }
            // Two proofs that one shorter one replaces, giving up a slot.
            let key = (63 << 58) | 100;
            assert!(proofs.add(key, proof(5, 40)));
            assert!(proofs.add(key, proof(3, 20)));
            assert!(proofs.add(key, proof(2, 60)));
            kept.push(key);
            assert_eq!(proofs.taken(), places.len() + 1);
            assert!(proofs.slots[64..].iter().any(|slot| slot.key() != FREE));
            assert_ordered(&proofs);
            assert_eq!(proofs.drop_mates(), mates.len());
            assert_eq!(proofs.taken(), kept.len());
            assert_ordered(&proofs);
            for &key in &kept {
                assert!(proofs.get(key, 0).is_some(), "{key:x}");
            }
            assert_eq!(proofs.get(key, 50).map(|p| p.length), Some(2));
            for &key in &mates {
                assert!(proofs.get(key, 0).is_none(), "{key:x}");
            }
            let mut keys: Vec<u64> = proofs.keys().collect();
            keys.sort_unstable();
            kept.sort_unstable();
            assert_eq!(keys, kept);
        }
    }

    /// A mate in one beats no other proof as short, and comes back as any
    /// other proof does; dropping a position's proofs leaves the others'.
    #[test]
    fn keeps_mates_as_other_proofs() {
        for shared in [false, true] {
            let proofs = Proofs::sized(16, shared);
            let mate = Proof {
                m: Some(Move::from_bits(0x0456)),
                length: 1,
                limit: 99,
            };
            let other = Proof {
                m: Some(Move::from_bits(0x0789)),
                ..mate
            };
            assert!(proofs.add(3, other));
            assert!(proofs.add_mate(3, mate));
            assert_eq!(proofs.get(3, 99), Some(other));
            assert!(proofs.add_mate(4, mate));
            assert!(proofs.add(4, other));
            assert!(proofs.add(4, proof(3, 99)));
            assert_eq!(proofs.get(4, 99), Some(mate));
            assert!(proofs.add(5, proof(4, 99)));
            proofs.remove(4);
            assert!(proofs.get(4, 0).is_none());
            assert_eq!(proofs.get(3, 0), Some(other));
            assert_eq!(proofs.get(5, 0).map(|p| p.length), Some(4));
            assert_eq!(proofs.drop_mates(), 0);
            assert_eq!(proofs.taken(), 2);
            assert_ordered(&proofs);
        }
    }

    /// A deleted proof's run moves down a slot up to the first proof at its
    /// own place, and every proof is found again.
    #[test]
    fn moves_runs_down_for_deletions() {
        for shared in [false, true] {
            let proofs = Proofs::sized(64, shared);
            // 64 places: a key's top six bits are its place. A run from place
            // 10 to slot 15, where the proof at place 15 sits at its own.
            let at = |place: u64, low: u64| (place << 58) | low;
            let keys = [
                at(10, 1),
                at(10, 2),
                at(10, 3),
                at(11, 4),
                at(13, 5),
                at(15, 6),
            ];
            for (i, &key) in keys.iter().enumerate() {
                assert!(proofs.add(key, proof(i as u32 + 1, 99)));
            }
            let slots = |proofs: &Proofs| -> Vec<u64> {
                proofs.slots[10..17].iter().map(|slot| slot.key()).collect()
            };
            assert_eq!(
                slots(&proofs),
                [keys[0], keys[1], keys[2], keys[3], keys[4], keys[5], FREE]
            );
            proofs.remove(keys[1]);
            assert_eq!(
                slots(&proofs),
                [keys[0], keys[2], keys[3], keys[4], FREE, keys[5], FREE]
            );
            assert_eq!(proofs.taken(), 5);
            assert_ordered(&proofs);
            // A proof that replaces two takes the first's slot, and the rest
            // of the run moves down into the second's.
            assert!(proofs.add(keys[3], proof(3, 40)));
            assert_eq!(
                slots(&proofs),
                [keys[0], keys[2], keys[3], keys[3], keys[4], keys[5], FREE]
            );
            assert!(proofs.add(keys[3], proof(2, 99)));
            assert_eq!(proofs.taken(), 5);
            assert!(proofs.add(keys[0], proof(0, 50)));
            assert_eq!(
                slots(&proofs),
                [keys[0], keys[0], keys[2], keys[3], keys[4], keys[5], FREE]
            );
            assert_eq!(proofs.taken(), 6);
            assert!(proofs.add(keys[0], proof(0, 99)));
            assert_eq!(
                slots(&proofs),
                [keys[0], keys[2], keys[3], keys[4], FREE, keys[5], FREE]
            );
            assert_eq!(proofs.taken(), 5);
            assert_ordered(&proofs);
            for (i, &key) in keys.iter().enumerate() {
                let length = match i {
                    0 => Some(0),
                    1 => None,
                    3 => Some(2),
                    _ => Some(i as u32 + 1),
                };
                assert_eq!(proofs.get(key, 0).map(|p| p.length), length, "{i}");
            }
        }
    }

    /// A key's proofs in this test: its move tells the key, and its length
    /// and limit one of three, so that a proof read half from one slot and
    /// half from another shows.
    fn key_proof(key: u64, which: u64) -> Proof {
        Proof {
            m: Some(Move::from_bits((key % 0x7ffe) as u16 + 1)),
            length: 10 - which as u32 * 3,
            limit: [40, 70, 99][which as usize],
        }
    }

    /// Threads that add and remove proofs of their own positions while all
    /// of them look up every position: a lookup finds only proofs added for
    /// its position, never misses a proof added before the threads started
    /// and never removed, and the table ends ordered, with each position's
    /// proofs as its thread left them. The keys share few places, so that
    /// runs are long and cross a stretch's end.
    #[test]
    fn threads_share_proofs() {
        let threads = 4u64;
        let proofs = Proofs::sized(2 * STRETCH, true);
        // The ith key, at one of 128 places from the 2,000th on, across the
        // first stretch's end. The threads' positions come first, each the
        // thread's whose number is its lowest bits, then the lasting ones.
        let band = |i: u64| (2000 + i % 128) * (u64::MAX / (2 * STRETCH as u64)) + 1 + i;
        let positions = 150 * threads;
        let lasting: Vec<u64> = (positions..positions + 100).map(band).collect();
        for &key in &lasting {
            assert!(proofs.add(key, key_proof(key, 0)));
        }
        let left: Vec<HashMap<u64, Option<u64>>> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|thread| {
                    let (proofs, lasting) = (&proofs, &lasting);
                    scope.spawn(move || {
                        let mut mine = HashMap::new();
                        let mut x = 0x2545_f491_4f6c_dd1d_u64 ^ thread;
                        for _ in 0..20_000 {
                            x ^= x << 13;
                            x ^= x >> 7;
                            x ^= x << 17;
                            let key = band((x % 150) * threads + thread);
                            match x >> 60 {
                                0..4 => {
                                    proofs.remove(key);
                                    mine.insert(key, None);
                                }
                                which => {
                                    let which = which % 3;
                                    assert!(proofs.add(key, key_proof(key, which)));
                                    let best = mine.get(&key).copied().flatten();
                                    mine.insert(
                                        key,
                                        Some(best.map_or(which, |b: u64| b.max(which))),
                                    );
                                }
                            }
                            let other = band((x >> 20) % positions);
                            if let Some(found) = proofs.get(other, 0) {
                                assert_eq!(found.m, key_proof(other, 0).m, "{other:x}");
                            }
                            let held = lasting[(x >> 40) as usize % lasting.len()];
                            assert_eq!(proofs.get(held, 0), Some(key_proof(held, 0)), "{held:x}");
                        }
                        mine
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect()
        });
        assert_ordered(&proofs);
        for mine in &left {
            for (&key, &best) in mine {
                assert_eq!(proofs.get(key, 0), best.map(|which| key_proof(key, which)));
            }
        }
        for &key in &lasting {
            assert_eq!(proofs.get(key, 0), Some(key_proof(key, 0)));
        }
    }
}
