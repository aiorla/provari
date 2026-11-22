//! Proven positions in a fixed amount of memory, in an ordered open-addressing
//! table (Amble and Knuth 1974): an entry sits at or after its key's place,
//! with every slot between them taken, and the keys of every run of taken
//! slots rise along it. A lookup must see each of a position's proofs, so it
//! reads from the key's place to the first key above its own, where an
//! unordered table reads to the end of the run: at nine tenths full, some 5
//! slots against 50. A key's place rises with the key, so a run in key order
//! keeps every entry at or after its place: an insertion moves the rest of its
//! run up a slot, and a deletion moves it down to the first proof at its own
//! place. A position keeps its shortest proof and any longer ones that hold at
//! higher counters.
//!
//! An entry takes 10 bytes. A position's key is its place and 40 bits of its
//! hash, compared in that order: the place comes from the hash's low half, and
//! the 40 bits are its high half, which the table's filter reads (`checks`),
//! and its lowest 8 bits. A place and 40 bits tell all hashes apart in a table
//! of 2^24 places or more (160 MB); in a smaller one, a lookup that misses
//! meets another position's proof at most once in 1.2 trillion. The entry
//! keeps the 40 bits and how far past its place it sits, which gives the place
//! back. At nine tenths full an entry sits 4.5 slots past its place on
//! average, each slot further about 0.8 times as likely: in a simulation the
//! most among 60 million entries was 69, so a billion stay near 100, far below
//! the 255 an entry can tell. An insertion that would take an entry further is
//! refused, as a full table's are. The rest of the entry is the winning move
//! where the attacker is to move, the proof's length in attacker moves and the
//! highest 50-move counter it holds at.
//!
//! Proofs are not dropped one by one: no line of a proof can repeat a position
//! because lengths shrink along it, and a proof found again after being
//! dropped could be longer than its parent assumed. So the table takes proofs
//! until it is nine tenths full, then refuses them; its lookups slow down as
//! it fills, which still beats stopping. Only proofs that come out the same
//! whenever they are worked out again are dropped, all at once. Mates in one
//! are marked, so that the search can drop them all when the table first fills
//! (`drop_mates`) and work each out again where it needs one; when it fills
//! again, it drops every proof of at most `SHORT` moves (`drop_short`), which
//! a search of that many moves finds again.
//!
//! Threads may share the table (`shared`). Its slots then fall into stretches
//! of `STRETCH`, each with a version: a thread that changes slots locks every
//! stretch they are in, in order, making their versions odd, and unlocks them
//! with the next even ones. A lookup reads the versions of the stretches it
//! walks before and after its walk, and walks again if one changed, so that
//! it never sees half a move of a run. Dropping the mates in one or the short
//! proofs, which moves every proof, waits until no other thread is in the
//! table.

use std::collections::TryReserveError;
use std::sync::atomic::{
    AtomicU16, AtomicU32, AtomicUsize,
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

/// A position's key: its place, then 40 bits of its hash, never all zero.
type Key = (usize, u64);

/// A proof as the table keeps it: the key, then in 32 bits the winning move in
/// 15 (0 for none, as no move goes from a square to itself), the length in 10
/// and the limit in 7: a counter up to `LAST`, `MATE` for a mate in one, which
/// holds up to `LAST`, or `ANY` for any.
#[derive(Clone, Copy)]
struct Entry {
    key: Key,
    rest: u32,
}

/// A slot, which holds an entry in 10 bytes, word by word, so that threads can
/// share the table: in three words its tag, the key's 40 bits and above them
/// how far past its place the slot is, then the rest in two.
#[derive(Default)]
#[repr(C)]
struct Slot {
    tag: [AtomicU16; 3],
    rest: [AtomicU16; 2],
}

const _: () = assert!(size_of::<Slot>() == 10);

/// The most attacker moves of a short proof: one the table drops with every
/// other when it fills after its mates in one are gone (`drop_short`), as a
/// search of so many moves works it out again.
pub const SHORT: u32 = 2;

const MOVE_BITS: u32 = 15;
const LENGTH_BITS: u32 = 10;
const LAST: u32 = 99;
const MATE: u32 = 100;
const ANY: u32 = 127;

/// The hash's bits a key keeps beside its place.
const KEPT_BITS: u32 = 40;
const KEPT: u64 = (1 << KEPT_BITS) - 1;

/// The most slots an entry may sit past its place: what a tag's top 8 bits
/// tell.
const MOST_PAST: usize = 255;

/// The tag of a free slot, as no key's bits are all zero.
const FREE: u64 = 0;

/// The slots of a stretch, which shares a version: 20 KB, so that the versions
/// of the largest tables still fit in a core's cache.
const STRETCH: usize = 1 << 11;

impl Entry {
    fn new(key: Key, proof: Proof, mate: bool) -> Entry {
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
    /// The key's 40 bits and above them how far past its place the slot is,
    /// `FREE` for none.
    fn tag(&self) -> u64 {
        u64::from(self.tag[0].load(Relaxed))
            | u64::from(self.tag[1].load(Relaxed)) << 16
            | u64::from(self.tag[2].load(Relaxed)) << 32
    }

    fn rest(&self) -> u32 {
        u32::from(self.rest[0].load(Relaxed)) | u32::from(self.rest[1].load(Relaxed)) << 16
    }

    /// The entry in this slot, the `index`th, unless the slot is free.
    fn load(&self, index: usize) -> Option<Entry> {
        let key = key_in(index, self.tag())?;
        Some(Entry {
            key,
            rest: self.rest(),
        })
    }

    /// Keeps an entry in this slot, the `index`th, which is at most
    /// `MOST_PAST` past its place.
    fn store(&self, index: usize, entry: Entry) {
        let (place, kept) = entry.key;
        debug_assert!(place <= index && index - place <= MOST_PAST && kept != FREE);
        let tag = kept | ((index - place) as u64) << KEPT_BITS;
        self.tag[0].store(tag as u16, Relaxed);
        self.tag[1].store((tag >> 16) as u16, Relaxed);
        self.tag[2].store((tag >> 32) as u16, Relaxed);
        self.rest[0].store(entry.rest as u16, Relaxed);
        self.rest[1].store((entry.rest >> 16) as u16, Relaxed);
    }

    /// Frees the slot, leaving the rest of its entry, which nothing reads.
    fn free(&self) {
        for word in &self.tag {
            word.store(0, Relaxed);
        }
    }
}

/// The key in the `index`th slot, from its tag, unless the slot is free. A
/// read torn by another thread's change may give any key, and its walk is
/// then made again.
fn key_in(index: usize, tag: u64) -> Option<Key> {
    (tag != FREE).then(|| {
        let past = (tag >> KEPT_BITS) as usize;
        (index.saturating_sub(past), tag & KEPT)
    })
}

// SAFETY: five atomic integers, which default to zero.
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
    /// past its last place (at most 2,560 bytes more), for threads to share
    /// or not, unless its memory cannot be had.
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

    /// Forgets every proof, as a new table of the same size.
    pub fn clear(&mut self) {
        crate::table::zero(&mut self.slots);
        crate::table::zero(&mut self.versions);
        *self.used.0.get_mut() = 0;
    }

    fn shared(&self) -> bool {
        !self.versions.is_empty()
    }

    /// The shortest proof of a position that holds at this counter.
    pub fn get(&self, hash: u64, counter: u32) -> Option<Proof> {
        let key = self.key(hash);
        if self.shared() {
            return self.read(key, counter);
        }
        let mut best: Option<Proof> = None;
        for (index, slot) in self.slots.iter().enumerate().skip(key.0) {
            let Some(held) = key_in(index, slot.tag()) else {
                break;
            };
            if held > key {
                break;
            }
            if held == key {
                let proof = Entry {
                    key,
                    rest: slot.rest(),
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
    fn read(&self, key: Key, counter: u32) -> Option<Proof> {
        let mut walks = 0u32;
        'walk: loop {
            walks += 1;
            wait(walks);
            // The stretches read and their versions; a run seldom crosses one
            // stretch's end, and one that crosses three is read locked.
            let mut read = [(0, 0); 4];
            let mut stretches = 0;
            let mut best: Option<Proof> = None;
            for (index, slot) in self.slots.iter().enumerate().skip(key.0) {
                let stretch = index / STRETCH;
                if stretches == 0 || read[stretches - 1].0 != stretch {
                    if stretches == read.len() {
                        return self.read_locked(key, counter);
                    }
                    let version = self.versions[stretch].load(Acquire);
                    if version & 1 != 0 {
                        continue 'walk;
                    }
                    read[stretches] = (stretch, version);
                    stretches += 1;
                }
                let Some(entry) = slot.load(index) else {
                    break;
                };
                if entry.key > key {
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
    fn read_locked(&self, key: Key, counter: u32) -> Option<Proof> {
        let mut locks = Locks::new(&self.versions, key.0);
        let mut best: Option<Proof> = None;
        let mut index = key.0;
        loop {
            let Some(entry) = self.slots[index].load(index) else {
                return best;
            };
            if entry.key > key {
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
        self.insert(Entry::new(self.key(hash), proof, false))
    }

    /// Adds a mate in one, as `add` does, marked for `drop_mates`.
    pub fn add_mate(&self, hash: u64, proof: Proof) -> bool {
        self.insert(Entry::new(self.key(hash), proof, true))
    }

    fn insert(&self, entry: Entry) -> bool {
        let key = entry.key;
        // Past the smaller keys of the run from the key's place to the
        // position's own proofs: a known one as good ends the search, and the
        // ones the new proof beats give up their slots, the first to it.
        let mut index = key.0;
        let mut locks = Locks::new(&self.versions, index);
        while let Some(held) = self.key_at(index)
            && held < key
        {
            index += 1;
            locks.reach(index);
        }
        let mut slot = None;
        while self.key_at(index) == Some(key) {
            let known = Entry {
                key,
                rest: self.slots[index].rest(),
            };
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
            self.slots[slot].store(slot, entry);
            return true;
        }
        // Else in its turn, the larger keys after it moving up a slot into
        // the free one that ends the run, while the table has room: a tenth
        // of it stays free, so runs end, and the last slot stays free; and
        // while no entry would sit more than `MOST_PAST` past its place.
        let mut past = index - key.0;
        let mut end = index;
        while let Some((place, _)) = self.key_at(end) {
            past = past.max(end + 1 - place);
            end += 1;
            locks.reach(end);
        }
        if past > MOST_PAST || end + 1 == self.slots.len() || !self.take() {
            return false;
        }
        for index in (index..end).rev() {
            self.shift(index, index + 1);
        }
        self.slots[index].store(index, entry);
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
        let key = self.key(hash);
        let mut index = key.0;
        let mut locks = Locks::new(&self.versions, index);
        while let Some(held) = self.key_at(index)
            && held < key
        {
            index += 1;
            locks.reach(index);
        }
        while self.key_at(index) == Some(key) {
            self.delete(index, &mut locks);
        }
    }

    /// Empties a taken slot, the rest of its run moving down a slot up to
    /// the first proof at its own place: the proofs after that one are at
    /// or past it, and so need no slot before it.
    fn delete(&self, index: usize, locks: &mut Locks) {
        let mut end = index + 1;
        locks.reach(end);
        while let Some((place, _)) = self.key_at(end)
            && place < end
        {
            end += 1;
            locks.reach(end);
        }
        for index in index + 1..end {
            self.shift(index, index - 1);
        }
        self.slots[end - 1].free();
        if self.shared() {
            self.used.fetch_sub(1, Relaxed);
        } else {
            self.used.store(self.used.load(Relaxed) - 1, Relaxed);
        }
    }

    /// Moves a taken slot's entry to another slot, which tells anew how far
    /// past its place it sits.
    fn shift(&self, from: usize, to: usize) {
        if let Some(entry) = self.slots[from].load(from) {
            self.slots[to].store(to, entry);
        }
    }

    /// Drops every mate in one and packs the rest down in place (`drop_where`).
    /// Returns how many mates it dropped. Shared, only while no other thread
    /// is in the table.
    pub fn drop_mates(&self) -> usize {
        self.drop_where(Entry::mate)
    }

    /// Drops every proof of at most `SHORT` moves and packs the rest down in
    /// place (`drop_where`). Returns how many it dropped. Shared, only while
    /// no other thread is in the table.
    pub fn drop_short(&self) -> usize {
        self.drop_where(|entry| entry.length() <= SHORT)
    }

    /// Drops the proofs `dropped` picks and packs the rest down in place: the
    /// slots' keys rise from the first to the last taken one, so each proof
    /// in turn moves to its key's place or the slot after the proof before
    /// it, whichever is later, at worst its own, and every run stays whole
    /// and in order. Returns how many it dropped.
    fn drop_where(&self, dropped: impl Fn(&Entry) -> bool) -> usize {
        let mut count = 0;
        let mut next = 0;
        for index in 0..self.slots.len() {
            let Some(entry) = self.slots[index].load(index) else {
                continue;
            };
            self.slots[index].free();
            if dropped(&entry) {
                count += 1;
                continue;
            }
            let slot = entry.key.0.max(next);
            self.slots[slot].store(slot, entry);
            next = slot + 1;
        }
        self.used.store(self.used.load(Relaxed) - count, Relaxed);
        count
    }

    /// Starts loading the first slot of a position's run (see
    /// `table::prefetch`), and the next one, whose line holds the end of the
    /// first where it crosses into another, and its stretch's version.
    pub fn prefetch(&self, hash: u64) {
        let start = self.key(hash).0;
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

    /// The filter's check of every position held (`table::check`), the low
    /// half of the bits its key keeps, for rebuilding the filter. Shared,
    /// only while no other thread is in the table.
    pub fn checks(&self) -> impl Iterator<Item = u32> + '_ {
        self.slots
            .iter()
            .map(Slot::tag)
            .filter(|&tag| tag != FREE)
            .map(|tag| (tag as u32).max(1))
    }

    /// A position's key, from its hash with its halves swapped: the place is
    /// that number's share of the places, by multiplying, which keeps the
    /// numbers' order, and the bits kept are its low 40, never all zero.
    fn key(&self, hash: u64) -> Key {
        let swapped = hash.rotate_left(32);
        let place = ((u128::from(swapped) * self.places as u128) >> 64) as usize;
        (place, (swapped & KEPT).max(1))
    }

    /// The key in a slot, unless it is free.
    fn key_at(&self, index: usize) -> Option<Key> {
        key_in(index, self.slots[index].tag())
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

/// The slots past the last place for the runs that go on from there, and the
/// free one that ends them: as many as an entry may sit past its place, or as
/// the table takes if fewer.
fn beyond(most: usize) -> usize {
    most.min(MOST_PAST) + 1
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::Ordering::Relaxed;

    use super::{FREE, KEPT_BITS, Key, MOST_PAST, Proof, Proofs, SHORT, STRETCH};
    use crate::board::Move;

    fn proof(length: u32, limit: u32) -> Proof {
        Proof {
            m: None,
            length,
            limit,
        }
    }

    /// The hash whose halves, swapped, are `swapped`: the tests choose keys
    /// by their places, the top bits of `swapped`.
    fn hash(swapped: u64) -> u64 {
        swapped.rotate_right(32)
    }

    /// How many slots hold proofs of the position.
    fn held(proofs: &Proofs, hash: u64) -> usize {
        let key = proofs.key(hash);
        (0..proofs.slots.len())
            .filter(|&index| proofs.key_at(index) == Some(key))
            .count()
    }

    /// Every taken slot's key at or after its place, at most `MOST_PAST`
    /// past it, with every slot between them taken, the keys rising along the
    /// table, the last slot free, and the slots taken counted; shared, every
    /// stretch unlocked.
    fn assert_ordered(proofs: &Proofs) {
        let mut last = (0, 0);
        let mut run = 0;
        let mut taken = 0;
        for index in 0..proofs.slots.len() {
            let Some(key) = proofs.key_at(index) else {
                run = index + 1;
                continue;
            };
            taken += 1;
            let past = (proofs.slots[index].tag() >> KEPT_BITS) as usize;
            assert!(
                run <= key.0 && key.0 + past == index && past <= MOST_PAST,
                "{index}: place {}, run from {run}",
                key.0
            );
            assert!(key >= last, "{index}: {key:?} after {last:?}");
            last = key;
        }
        assert_eq!(proofs.slots.last().map(|slot| slot.tag()), Some(FREE));
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
            assert_eq!(held(&proofs, 7), 1);
            // A longer one that holds less far adds nothing.
            assert!(proofs.add(7, proof(6, 50)));
            assert_eq!(held(&proofs, 7), 1);
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
                .map(|k| hash(k.wrapping_mul(1 << 61) | k))
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
    /// the stretches of a shared table and past the last place.
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
                // One in forty in the last sixty-fourth of the places, so
                // that runs go on past the last place.
                let kept = hash(if added.len() % 40 == 0 {
                    key | 0xfc00 << 48
                } else {
                    key
                });
                assert!(proofs.add(kept, proof(added.len() as u32 % 1000 + 1, 99)));
                added.push(kept);
            }
            assert!(!proofs.add(hash(key ^ 1), proof(1, 99)));
            assert_ordered(&proofs);
            assert!(proofs.slots[places..].iter().any(|slot| slot.tag() != FREE));
            for (i, &kept) in added.iter().enumerate() {
                let found = proofs.get(kept, 0);
                assert_eq!(found.map(|p| p.length), Some(i as u32 % 1000 + 1));
                if shared {
                    // As a lookup that crosses too many stretches reads it.
                    assert_eq!(proofs.read_locked(proofs.key(kept), 0), found);
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
            // 64 places: a key's place is the top six bits of its swapped
            // hash. Places near the end, where the runs go on past it, and at
            // the start.
            let places = [62, 63, 63, 61, 0, 63, 1, 62, 0, 2, 63, 60, 1, 3, 62, 63];
            let mut kept = Vec::new();
            let mut mates = Vec::new();
            for (i, &place) in places.iter().enumerate() {
                let key = hash((place << 58) | (i as u64 + 1));
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
            let key = hash((63 << 58) | 100);
            assert!(proofs.add(key, proof(5, 40)));
            assert!(proofs.add(key, proof(3, 20)));
            assert!(proofs.add(key, proof(2, 60)));
            kept.push(key);
            assert_eq!(proofs.taken(), places.len() + 1);
            assert!(proofs.slots[64..].iter().any(|slot| slot.tag() != FREE));
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
            let mut keys: Vec<Key> = (0..proofs.slots.len())
                .filter_map(|index| proofs.key_at(index))
                .collect();
            let mut kept: Vec<Key> = kept.iter().map(|&key| proofs.key(key)).collect();
            keys.sort_unstable();
            kept.sort_unstable();
            assert_eq!(keys, kept);
        }
    }

    /// Dropping the short proofs leaves every longer one where a lookup finds
    /// it, in runs that share slots and go on past the last place, and frees
    /// their slots: a tablebase win, a mate in one, marked or not, and a win
    /// in two go, and of a position's proofs the longer one that holds
    /// further stays.
    #[test]
    fn drops_short_proofs_and_keeps_the_rest() {
        for shared in [false, true] {
            let proofs = Proofs::sized(64, shared);
            // 64 places: a key's place is the top six bits of its swapped
            // hash. Places near the end, where the runs go on past it, and at
            // the start.
            let places = [62, 63, 63, 61, 0, 63, 1, 62, 0, 2, 63, 60, 1, 3, 62, 63];
            let mut kept = Vec::new();
            let mut short = Vec::new();
            for (i, &place) in places.iter().enumerate() {
                let key = hash((place << 58) | (i as u64 + 1));
                // Lengths 0 to 5 in turn, every other mate in one marked.
                let length = i as u32 % 6;
                if length == 1 && i % 4 == 1 {
                    let mate = Proof {
                        m: Some(Move::from_bits(0x0123)),
                        length: 1,
                        limit: 99,
                    };
                    assert!(proofs.add_mate(key, mate));
                } else {
                    assert!(proofs.add(key, proof(length, 99)));
                }
                if length <= SHORT {
                    short.push(key);
                } else {
                    kept.push(key);
                }
            }
            // A position's short proof, and its longer one that holds further.
            let key = hash((63 << 58) | 100);
            assert!(proofs.add(key, proof(2, 20)));
            assert!(proofs.add(key, proof(5, 60)));
            kept.push(key);
            assert_eq!(proofs.get(key, 10).map(|p| p.length), Some(2));
            assert_eq!(proofs.taken(), places.len() + 2);
            assert!(proofs.slots[64..].iter().any(|slot| slot.tag() != FREE));
            assert_ordered(&proofs);
            assert_eq!(proofs.drop_short(), short.len() + 1);
            assert_eq!(proofs.taken(), kept.len());
            assert_ordered(&proofs);
            for &key in &kept {
                assert!(proofs.get(key, 0).is_some(), "{key:x}");
            }
            assert_eq!(proofs.get(key, 10).map(|p| p.length), Some(5));
            for &key in &short {
                assert!(proofs.get(key, 0).is_none(), "{key:x}");
            }
            let mut keys: Vec<Key> = (0..proofs.slots.len())
                .filter_map(|index| proofs.key_at(index))
                .collect();
            let mut kept: Vec<Key> = kept.iter().map(|&key| proofs.key(key)).collect();
            keys.sort_unstable();
            kept.sort_unstable();
            assert_eq!(keys, kept);
            assert_eq!(proofs.drop_short(), 0);
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
            // 64 places: a key's place is the top six bits of its swapped
            // hash. A run from place 10 to slot 15, where the proof at place
            // 15 sits at its own.
            let at = |place: u64, low: u64| hash((place << 58) | low);
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
            let slots = |proofs: &Proofs| -> Vec<Option<Key>> {
                (10..17).map(|index| proofs.key_at(index)).collect()
            };
            let k = |i: usize| Some(proofs.key(keys[i]));
            assert_eq!(slots(&proofs), [k(0), k(1), k(2), k(3), k(4), k(5), None]);
            proofs.remove(keys[1]);
            assert_eq!(slots(&proofs), [k(0), k(2), k(3), k(4), None, k(5), None]);
            assert_eq!(proofs.taken(), 5);
            assert_ordered(&proofs);
            // A proof that replaces two takes the first's slot, and the rest
            // of the run moves down into the second's.
            assert!(proofs.add(keys[3], proof(3, 40)));
            assert_eq!(slots(&proofs), [k(0), k(2), k(3), k(3), k(4), k(5), None]);
            assert!(proofs.add(keys[3], proof(2, 99)));
            assert_eq!(proofs.taken(), 5);
            assert!(proofs.add(keys[0], proof(0, 50)));
            assert_eq!(slots(&proofs), [k(0), k(0), k(2), k(3), k(4), k(5), None]);
            assert_eq!(proofs.taken(), 6);
            assert!(proofs.add(keys[0], proof(0, 99)));
            assert_eq!(slots(&proofs), [k(0), k(2), k(3), k(4), None, k(5), None]);
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

    /// An insertion that would take an entry more than `MOST_PAST` slots past
    /// its place is refused, as a full table's are, and changes nothing.
    #[test]
    fn refuses_entries_too_far_past_their_place() {
        for shared in [false, true] {
            let proofs = Proofs::sized(1024, shared);
            // 1,024 places: a key's place is the top ten bits of its swapped
            // hash. 256 keys at place 5 fill slots 5 to 260, the last 255
            // past it.
            let at = |place: u64, low: u64| hash((place << 54) | low);
            for low in 0..=MOST_PAST as u64 {
                assert!(proofs.add(at(5, 1000 + low), proof(2, 99)));
            }
            // One more after them, or one before them that moves them all
            // up, would sit 256 past.
            assert!(!proofs.add(at(5, 2000), proof(2, 99)));
            assert!(!proofs.add(at(5, 1), proof(2, 99)));
            // Place 6's first proof sits 255 past it, its second 256.
            assert!(proofs.add(at(6, 1), proof(2, 99)));
            assert!(!proofs.add(at(6, 2), proof(2, 99)));
            // A better proof of a position held takes its slot.
            assert!(proofs.add(at(5, 1000), proof(1, 99)));
            assert_eq!(proofs.taken(), MOST_PAST + 2);
            assert_ordered(&proofs);
            for low in 0..=MOST_PAST as u64 {
                assert!(proofs.get(at(5, 1000 + low), 0).is_some());
            }
            assert_eq!(proofs.get(at(5, 1000), 0).map(|p| p.length), Some(1));
            assert!(proofs.get(at(5, 2000), 0).is_none() && proofs.get(at(5, 1), 0).is_none());
            assert!(proofs.get(at(6, 2), 0).is_none());
            // Elsewhere the table takes proofs as before.
            assert!(proofs.add(at(400, 7), proof(2, 99)));
            assert_ordered(&proofs);
        }
    }

    /// The checks of the positions held are the filter's (`table::check`).
    #[test]
    fn gives_the_filters_checks() {
        let proofs = Proofs::sized(1024, false);
        let hashes = [
            0,
            1,
            5 << 32,
            u64::MAX,
            0x0123_4567_89ab_cdef,
            0xfedc_ba98_0000_0000,
        ];
        for &hash in &hashes {
            assert!(proofs.add(hash, proof(3, 99)));
        }
        let mut checks: Vec<u32> = proofs.checks().collect();
        let mut expected: Vec<u32> = hashes.iter().map(|&h| crate::table::check(h)).collect();
        checks.sort_unstable();
        expected.sort_unstable();
        assert_eq!(checks, expected);
    }

    /// Random additions, removals and drops of mates, on positions crowded
    /// at a few places and spread elsewhere, give the answers of a plain map
    /// that keeps the same proofs, while the table stays ordered.
    #[test]
    fn answers_as_a_map_would() {
        /// A position's proofs as the map keeps them, each marked if a mate.
        type Kept = HashMap<u64, Vec<(Proof, bool)>>;
        fn add(kept: &mut Kept, hash: u64, new: Proof, mate: bool) {
            let known = kept.entry(hash).or_default();
            if known
                .iter()
                .any(|(old, _)| old.length <= new.length && old.limit >= new.limit)
            {
                return;
            }
            known.retain(|(old, _)| !(old.length >= new.length && old.limit <= new.limit));
            known.push((new, mate));
        }
        fn get(kept: &Kept, hash: u64, counter: u32) -> Option<Proof> {
            kept.get(&hash)?
                .iter()
                .map(|&(proof, _)| proof)
                .filter(|proof| proof.limit >= counter)
                .min_by_key(|proof| proof.length)
        }
        for (places, shared) in [(2048, false), (2048, true), (3 * STRETCH, true)] {
            let proofs = Proofs::sized(places, shared);
            let mut kept = Kept::new();
            // Additions refused by a full table, and those that would take
            // an entry too far past its place.
            let mut refused = [0; 2];
            let mut x = 0x2545_f491_4f6c_dd1d_u64 ^ places as u64;
            let mut next = || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            };
            // Positions: a tenth at each of two places side by side and a
            // tenth at the last, so that their runs reach the most an entry
            // may sit past its place, and the rest spread, so that the table
            // fills.
            let width = u64::MAX / places as u64;
            let count = places * 3 / 2;
            let positions: Vec<u64> = (0..count)
                .map(|i| {
                    let place = match i % 10 {
                        0 => places as u64 / 7,
                        1 => places as u64 / 7 + 1,
                        2 => places as u64 - 1,
                        _ => next() % places as u64,
                    };
                    hash(place * width + next() % width)
                })
                .collect();
            for step in 0..200_000u32 {
                let r = next();
                let hash = positions[(r % count as u64) as usize];
                match (r >> 20) % 100 {
                    0..60 => {
                        let limit = match (r >> 30) % 102 {
                            100 | 101 => u32::MAX,
                            limit => limit as u32,
                        };
                        let new = Proof {
                            m: Some(Move::from_bits(((r >> 40) % 0x7fff + 1) as u16)),
                            length: (r >> 50) as u32 % 40,
                            limit,
                        };
                        if proofs.add(hash, new) {
                            add(&mut kept, hash, new, false);
                        } else {
                            refused[usize::from(proofs.taken() < proofs.room())] += 1;
                        }
                    }
                    60..65 => {
                        let mate = Proof {
                            m: Some(Move::from_bits(((r >> 40) % 0x7fff + 1) as u16)),
                            length: 1,
                            limit: 99,
                        };
                        if proofs.add_mate(hash, mate) {
                            add(&mut kept, hash, mate, true);
                        } else {
                            refused[usize::from(proofs.taken() < proofs.room())] += 1;
                        }
                    }
                    65..80 => {
                        proofs.remove(hash);
                        kept.remove(&hash);
                    }
                    _ => {
                        let counter = ((r >> 30) % 101) as u32;
                        assert_eq!(proofs.get(hash, counter), get(&kept, hash, counter));
                    }
                }
                if step % 20_000 == 19_999 {
                    if step % 40_000 == 39_999 {
                        let dropped = proofs.drop_mates();
                        let mut mates = 0;
                        for known in kept.values_mut() {
                            mates += known.iter().filter(|(_, mate)| *mate).count();
                            known.retain(|(_, mate)| !mate);
                        }
                        assert_eq!(dropped, mates);
                    }
                    assert_ordered(&proofs);
                    let held: usize = kept.values().map(Vec::len).sum();
                    assert_eq!(proofs.taken(), held);
                    for &hash in &positions {
                        for counter in [0, 50, 99, 100] {
                            assert_eq!(proofs.get(hash, counter), get(&kept, hash, counter));
                        }
                    }
                }
            }
            assert!(refused.iter().all(|&count| count > 0), "{refused:?}");
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
        // The ith position, at one of 640 places from the 1,700th on, across
        // the first stretch's end. The threads' positions come first, each
        // the thread's whose number is its lowest bits, then the lasting
        // ones.
        let band = |i: u64| hash((1700 + i % 640) * (u64::MAX / (2 * STRETCH as u64)) + 1 + i);
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
