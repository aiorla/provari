//! Proven positions in a fixed amount of memory. An entry keeps a position's
//! whole key, the winning move where the attacker is to move, the proof's
//! length in attacker moves and the highest 50-move counter it holds at, in 12
//! bytes, in an open-addressing table: an entry sits at the first free slot
//! from its key's place on. A position keeps its shortest proof and any longer
//! ones that hold at higher counters.
//!
//! Proofs are not dropped one by one: no line of a proof can repeat a position
//! because lengths shrink along it, and a proof found again after being
//! dropped could be longer than its parent assumed. So the table takes proofs
//! until it is nine tenths full, then refuses them; its lookups slow down as
//! it fills, which still beats stopping. Mates in one are marked, so that the
//! search can drop them all when the table first fills (`drop_mates`) and work
//! each out again where it needs one.

use std::collections::TryReserveError;

use crate::board::Move;

/// A proven position's winning move where the attacker is to move, its length
/// in attacker moves, and the highest 50-move counter it holds at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Proof {
    pub m: Option<Move>,
    pub length: u32,
    pub limit: u32,
}

/// A proof as the table keeps it, in 12 bytes: the key, then in 32 bits the
/// winning move in 15 (0 for none, as no move goes from a square to itself),
/// the length in 10 and the limit in 7: a counter up to `LAST`, `MATE` for a
/// mate in one, which holds up to `LAST`, or `ANY` for any.
#[derive(Clone, Copy, Default)]
#[repr(C, packed(4))]
struct Entry {
    key: u64,
    rest: u32,
}

const _: () = assert!(size_of::<Entry>() == 12);

const MOVE_BITS: u32 = 15;
const LENGTH_BITS: u32 = 10;
const LAST: u32 = 99;
const MATE: u32 = 100;
const ANY: u32 = 127;

/// The key of a free slot, and of a slot whose proof a better one replaced;
/// positions with those keys are kept under others.
const FREE: u64 = 0;
const REPLACED: u64 = u64::MAX;

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

    fn key(&self) -> u64 {
        self.key
    }

    fn length(&self) -> u32 {
        let rest = self.rest;
        (rest >> MOVE_BITS) & ((1 << LENGTH_BITS) - 1)
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
        let rest = self.rest;
        rest >> (MOVE_BITS + LENGTH_BITS)
    }

    fn proof(&self) -> Proof {
        let rest = self.rest;
        let m = (rest & ((1 << MOVE_BITS) - 1)) as u16;
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

pub struct Proofs {
    entries: Vec<Entry>,
    /// Slots taken, replaced ones included, and the most there may be.
    used: usize,
    most: usize,
}

impl Proofs {
    /// A table of at most `megabytes`, and at least a few entries, unless its
    /// memory cannot be had.
    pub fn new(megabytes: usize) -> Result<Proofs, TryReserveError> {
        let entries = (megabytes.saturating_mul(1 << 20) / size_of::<Entry>()).max(4);
        Ok(Proofs {
            entries: crate::table::large(entries)?,
            used: 0,
            most: entries - (entries / 10).max(1),
        })
    }

    /// A table of `size` slots, for tests.
    #[cfg(test)]
    pub fn sized(size: usize) -> Proofs {
        Proofs {
            entries: vec![Entry::default(); size],
            used: 0,
            most: size - (size / 10).max(1),
        }
    }

    /// The shortest proof of a position that holds at this counter.
    pub fn get(&self, hash: u64, counter: u32) -> Option<Proof> {
        let key = key(hash);
        self.run(key)
            .filter(|&index| self.entries[index].key() == key)
            .map(|index| self.entries[index].proof())
            .filter(|proof| proof.limit >= counter)
            .min_by_key(|proof| proof.length)
    }

    /// Adds a proof unless a known one is as short and holds as far, and drops
    /// the known ones it beats on both. Returns false if the table is too full
    /// to take it.
    pub fn add(&mut self, hash: u64, proof: Proof) -> bool {
        self.insert(Entry::new(key(hash), proof, false))
    }

    /// Adds a mate in one, as `add` does, marked for `drop_mates`.
    pub fn add_mate(&mut self, hash: u64, proof: Proof) -> bool {
        self.insert(Entry::new(key(hash), proof, true))
    }

    fn insert(&mut self, entry: Entry) -> bool {
        let key = entry.key();
        // The run of taken slots from the key's place: a known proof as good
        // ends the search, ones the new proof beats give up their slots.
        let mut index = self.start(key);
        let mut slot = None;
        while self.entries[index].key() != FREE {
            let known = self.entries[index];
            if known.key() == key {
                if known.length() <= entry.length() && known.limit() >= entry.limit() {
                    return true;
                }
                if known.length() >= entry.length() && known.limit() <= entry.limit() {
                    self.entries[index].key = REPLACED;
                }
            }
            if self.entries[index].key() == REPLACED && slot.is_none() {
                slot = Some(index);
            }
            index = self.next(index);
        }
        // Else the free slot ending the run, while the table has room: a
        // tenth of it stays free, so runs end.
        let index = match slot {
            Some(slot) => slot,
            None if self.used < self.most => {
                self.used += 1;
                index
            }
            None => return false,
        };
        self.entries[index] = entry;
        true
    }

    /// Drops a position's proofs.
    pub fn remove(&mut self, hash: u64) {
        let key = key(hash);
        let mut index = self.start(key);
        while self.entries[index].key() != FREE {
            if self.entries[index].key() == key {
                self.entries[index].key = REPLACED;
            }
            index = self.next(index);
        }
    }

    /// Drops every mate in one, and the slots of replaced proofs, and packs
    /// the rest back in place: from a slot free before, which no run crosses,
    /// each proof in turn moves back to the first free slot from its key's
    /// place, at worst its own, so that every run stays whole. Returns how
    /// many mates it dropped.
    pub fn drop_mates(&mut self) -> usize {
        let Some(free) = self.entries.iter().position(|entry| entry.key() == FREE) else {
            return 0;
        };
        let mut dropped = 0;
        self.used = 0;
        let mut index = free;
        for _ in 1..self.entries.len() {
            index = self.next(index);
            let entry = self.entries[index];
            if entry.key() == FREE {
                continue;
            }
            self.entries[index].key = FREE;
            if entry.key() == REPLACED {
                continue;
            }
            if entry.mate() {
                dropped += 1;
                continue;
            }
            let mut slot = self.start(entry.key());
            while self.entries[slot].key() != FREE {
                slot = self.next(slot);
            }
            self.entries[slot] = entry;
            self.used += 1;
        }
        dropped
    }

    /// Starts loading the first slot of a position's run (see
    /// `table::prefetch`), and the next one, whose line holds the end of the
    /// first where it crosses into another.
    pub fn prefetch(&self, hash: u64) {
        let start = self.start(key(hash));
        crate::table::prefetch(&self.entries[start]);
        if let Some(next) = self.entries.get(start + 1) {
            crate::table::prefetch(next);
        }
    }

    /// The slots taken, replaced ones included: at least the positions held.
    pub fn taken(&self) -> usize {
        self.used
    }

    /// How full the table is, as a share of the most it takes.
    pub fn fill(&self) -> f64 {
        self.used as f64 / self.most as f64
    }

    /// The most slots it takes.
    pub fn room(&self) -> usize {
        self.most
    }

    /// The keys of every position held, for rebuilding a filter.
    pub fn keys(&self) -> impl Iterator<Item = u64> + '_ {
        self.entries
            .iter()
            .map(Entry::key)
            .filter(|&key| key != FREE && key != REPLACED)
    }

    /// A key's place: any number of slots, by multiplying.
    fn start(&self, key: u64) -> usize {
        ((u128::from(key) * self.entries.len() as u128) >> 64) as usize
    }

    /// The slot after `index`, the first after the last.
    fn next(&self, index: usize) -> usize {
        if index + 1 == self.entries.len() {
            0
        } else {
            index + 1
        }
    }

    /// The slots a key's entries may be in: from its place on, up to the first
    /// free slot.
    fn run(&self, key: u64) -> impl Iterator<Item = usize> + '_ {
        let size = self.entries.len();
        let start = self.start(key);
        (start..size)
            .chain(0..start)
            .take_while(|&index| self.entries[index].key() != FREE)
    }
}

fn key(hash: u64) -> u64 {
    match hash {
        FREE => 1,
        REPLACED => REPLACED - 1,
        hash => hash,
    }
}

#[cfg(test)]
mod tests {
    use super::{Proof, Proofs};
    use crate::board::Move;

    fn proof(length: u32, limit: u32) -> Proof {
        Proof {
            m: None,
            length,
            limit,
        }
    }

    /// The shortest proof that holds at the counter, among a position's
    /// proofs at different counters.
    #[test]
    fn keeps_the_shortest_proof_for_each_counter() {
        let mut proofs = Proofs::new(1).unwrap();
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
    }

    /// A proof comes back as it went in: every bit a move uses, lengths up to
    /// 1,023 and limits up to 99 or any.
    #[test]
    fn keeps_proofs_whole() {
        let mut proofs = Proofs::new(1).unwrap();
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

    /// Positions whose keys start at the same slot all keep their proofs, and
    /// the table refuses proofs past nine tenths full.
    #[test]
    fn fills_to_nine_tenths() {
        let mut proofs = Proofs::new(0).unwrap();
        let size = proofs.entries.len();
        // Keys spread over the slots, and some that share them.
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
    }

    /// Dropping the mates in one, and the slots of replaced proofs, leaves
    /// every other proof where a lookup finds it, in runs that share slots
    /// and cross the end of the table, and frees their slots.
    #[test]
    fn drops_mates_and_keeps_the_rest() {
        let mut proofs = Proofs::sized(64);
        // 64 slots: a key's top six bits are its place. Places near the end,
        // and at the start where the runs from the end go on.
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
        // Two proofs that one shorter one replaces, leaving a slot replaced.
        let key = (63 << 58) | 100;
        assert!(proofs.add(key, proof(5, 40)));
        assert!(proofs.add(key, proof(3, 20)));
        assert!(proofs.add(key, proof(2, 60)));
        kept.push(key);
        assert_eq!(proofs.taken(), places.len() + 2);
        assert_eq!(proofs.drop_mates(), mates.len());
        assert_eq!(proofs.taken(), kept.len());
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

    /// A mate in one beats no other proof as short, and comes back as any
    /// other proof does; dropping a position's proofs leaves the others'.
    #[test]
    fn keeps_mates_as_other_proofs() {
        let mut proofs = Proofs::sized(16);
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
    }
}
