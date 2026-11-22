//! A table's bytes: tolius's files are dictzip's (gzip whose data is
//! deflated in chunks of some 58 KB, each of which inflates on its own, with
//! their lengths in the header), or a table that `unpack` wrote whole, with
//! a header of our own. Both are mapped into memory and read in place: a
//! packed table's chunks are inflated as they are read, and kept for good
//! while the set's limit allows, else among the last few that each thread
//! inflated.

use std::cell::RefCell;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering::Relaxed};

use memmap2::Mmap;
use miniz_oxide::inflate::TINFLStatus;
use miniz_oxide::inflate::core::{DecompressorOxide, decompress, inflate_flags};

/// An unpacked table's first bytes, then the gzip header's MTIME, whose low
/// 16 bits are the table's flags, the CRC-32 of its bytes and their length,
/// all little-endian, then the bytes (`DICT_PLAIN_MAGIC`).
const PLAIN_MAGIC: &[u8; 8] = b"an0plain";
const PLAIN_HEADER: usize = 24;

/// The chunks each thread keeps of those it inflated beyond the set's limit.
const RECENT: usize = 8;

/// What a set's tables keep of their inflated chunks, over all of them.
#[derive(Default)]
pub struct Kept {
    limit: AtomicUsize,
    bytes: AtomicUsize,
}

impl Kept {
    pub fn set_limit(&self, bytes: usize) {
        self.limit.store(bytes, Relaxed);
    }

    pub fn bytes(&self) -> usize {
        self.bytes.load(Relaxed)
    }

    /// Takes `bytes` more if they fit within the limit. Once the limit is
    /// reached a plain load answers, so that threads do not contend for the
    /// count.
    fn take(&self, bytes: usize) -> bool {
        let limit = self.limit.load(Relaxed);
        if self.bytes.load(Relaxed).saturating_add(bytes) > limit {
            return false;
        }
        if self.bytes.fetch_add(bytes, Relaxed).saturating_add(bytes) <= limit {
            return true;
        }
        self.bytes.fetch_sub(bytes, Relaxed);
        false
    }

    fn give_back(&self, bytes: usize) {
        self.bytes.fetch_sub(bytes, Relaxed);
    }
}

/// A table file, open for reading.
pub struct TableFile {
    map: Mmap,
    /// The gzip header's MTIME, whose low 16 bits are the table's flags.
    pub mtime: u32,
    /// The CRC-32 of the table's bytes, from gzip's trailer or our header.
    pub crc: u32,
    /// The table's bytes.
    pub length: u64,
    kind: Kind,
}

enum Kind {
    Plain,
    Packed(Packed),
}

struct Packed {
    /// Set apart for each file, so that threads' recent chunks name it.
    id: u32,
    chunk_length: usize,
    /// Where each chunk starts in the file, and where the last ends.
    offsets: Box<[usize]>,
    kept: Box<[OnceLock<Box<[u8]>>]>,
}

static NEXT_ID: AtomicU32 = AtomicU32::new(0);

/// A thread's last inflated chunks: its file, its chunk, when last read, the
/// bytes.
struct Recent {
    clock: u64,
    slots: Vec<(u32, u32, u64, Box<[u8]>)>,
}

thread_local! {
    static RECENT_CHUNKS: RefCell<Recent> = const {
        RefCell::new(Recent { clock: 0, slots: Vec::new() })
    };
}

fn le(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .rev()
        .fold(0, |sum, &b| sum << 8 | u64::from(b))
}

impl TableFile {
    /// Opens a packed table or one `unpack` wrote.
    pub fn open(path: &Path) -> Result<TableFile, String> {
        let failed = |error: std::io::Error| format!("{}: {error}", path.display());
        let file = File::open(path).map_err(failed)?;
        // SAFETY: the tables are read only, by this process and others; one
        // changed under a search would be read wrong, as any file would.
        let map = unsafe { Mmap::map(&file) }.map_err(failed)?;
        let refuse = |why: &str| format!("{}: {why}", path.display());
        if map.len() >= 2 && map[..2] == [0x1f, 0x8b] {
            Self::packed(map).map_err(|why| refuse(&why))
        } else if map.len() >= PLAIN_HEADER && &map[..8] == PLAIN_MAGIC {
            let length = le(&map[16..24]);
            if map.len() as u64 != PLAIN_HEADER as u64 + length {
                return Err(refuse(&format!(
                    "not the {} bytes of its header and its table",
                    PLAIN_HEADER as u64 + length
                )));
            }
            Ok(TableFile {
                mtime: le(&map[8..12]) as u32,
                crc: le(&map[12..16]) as u32,
                length,
                kind: Kind::Plain,
                map,
            })
        } else {
            Err(refuse("neither a dzip file nor an unpacked table"))
        }
    }

    /// Reads dictzip's header: gzip's, its extra field holding the chunks'
    /// length and each one's packed bytes.
    fn packed(map: Mmap) -> Result<TableFile, String> {
        let bytes = &map[..];
        let short = || "a header cut short".to_owned();
        let at = |i: usize, n: usize| bytes.get(i..i + n).map(le).ok_or_else(short);
        let flags = at(3, 1)?;
        let mtime = at(4, 4)? as u32;
        if at(2, 1)? != 8 || flags & 4 == 0 || at(12, 2)? != u64::from(u16::from_le_bytes(*b"RA")) {
            return Err("not a dzip file".into());
        }
        let extra = at(10, 2)? as usize;
        let (version, chunk_length, count) = (at(16, 2)?, at(18, 2)? as usize, at(20, 2)? as usize);
        if version != 1 || chunk_length == 0 || count == 0 || 22 + 2 * count != 12 + extra {
            return Err("a dzip header of another kind".into());
        }
        let mut start = 12 + extra;
        for flag in [8, 16] {
            // The original name, then a comment, each ending in NUL.
            if flags & flag != 0 {
                let end = bytes[start..]
                    .iter()
                    .position(|&b| b == 0)
                    .ok_or_else(short)?;
                start += end + 1;
            }
        }
        if flags & 2 != 0 {
            start += 2;
        }
        if bytes.len() < start + 8 {
            return Err(short());
        }
        let crc = at(bytes.len() - 8, 4)? as u32;
        let length = at(bytes.len() - 4, 4)?;
        let mut offsets = Vec::with_capacity(count + 1);
        offsets.push(start);
        for i in 0..count {
            let end = offsets[i] + at(22 + 2 * i, 2)? as usize;
            offsets.push(end);
        }
        if offsets[count] > bytes.len() - 8
            || (count as u64 - 1) * chunk_length as u64 >= length
            || (count as u64) * (chunk_length as u64) < length
        {
            return Err("chunks that do not fit the file".into());
        }
        Ok(TableFile {
            mtime,
            crc,
            length,
            kind: Kind::Packed(Packed {
                id: NEXT_ID.fetch_add(1, Relaxed),
                chunk_length,
                offsets: offsets.into(),
                kept: (0..count).map(|_| OnceLock::new()).collect(),
            }),
            map,
        })
    }

    pub fn is_plain(&self) -> bool {
        matches!(self.kind, Kind::Plain)
    }

    /// The table's flags (`dz_get_flags`).
    pub fn flags(&self) -> u16 {
        self.mtime as u16
    }

    /// Reads the table's bytes from `start` into `out`, which must lie within
    /// them.
    pub fn read(&self, start: u64, out: &mut [u8], kept: &Kept) -> Result<(), String> {
        let end = start + out.len() as u64;
        if end > self.length {
            return Err("a read beyond the table".into());
        }
        match &self.kind {
            Kind::Plain => {
                let from = PLAIN_HEADER + start as usize;
                out.copy_from_slice(&self.map[from..from + out.len()]);
                Ok(())
            }
            Kind::Packed(packed) => {
                let length = packed.chunk_length as u64;
                let mut done = 0;
                while done < out.len() {
                    let at = start + done as u64;
                    let chunk = (at / length) as usize;
                    let offset = (at % length) as usize;
                    done += self.with_chunk(packed, chunk, kept, |bytes| {
                        let n = (bytes.len() - offset).min(out.len() - done);
                        out[done..done + n].copy_from_slice(&bytes[offset..offset + n]);
                        n
                    })?;
                }
                Ok(())
            }
        }
    }

    /// The unpacked length of chunk `i`, the last one shorter.
    fn chunk_length(&self, packed: &Packed, i: usize) -> usize {
        if i + 1 == packed.kept.len() {
            (self.length - i as u64 * packed.chunk_length as u64) as usize
        } else {
            packed.chunk_length
        }
    }

    /// Inflates chunk `i` into `out`, a byte longer than the chunk, so that
    /// a chunk inflating to more cannot pass for one that ends.
    fn inflate(&self, packed: &Packed, i: usize, out: &mut [u8]) -> Result<(), String> {
        let length = self.chunk_length(packed, i);
        let input = &self.map[packed.offsets[i]..packed.offsets[i + 1]];
        let mut state = DecompressorOxide::new();
        let flags = inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF
            | inflate_flags::TINFL_FLAG_HAS_MORE_INPUT;
        let (status, read, written) = decompress(&mut state, input, &mut out[..=length], 0, flags);
        let finished = matches!(status, TINFLStatus::Done | TINFLStatus::NeedsMoreInput);
        if !finished || read != input.len() || written != length {
            return Err(format!("chunk {i} does not inflate to its {length} bytes"));
        }
        Ok(())
    }

    /// Calls `f` with chunk `i`'s bytes: kept, recent, or inflated now, and
    /// then kept if the limit allows.
    fn with_chunk<T>(
        &self,
        packed: &Packed,
        i: usize,
        kept: &Kept,
        f: impl FnOnce(&[u8]) -> T,
    ) -> Result<T, String> {
        if let Some(bytes) = packed.kept[i].get() {
            return Ok(f(bytes));
        }
        let length = self.chunk_length(packed, i);
        RECENT_CHUNKS.with(|recent| {
            let mut recent = recent.borrow_mut();
            let recent = &mut *recent;
            recent.clock += 1;
            let clock = recent.clock;
            let (id, chunk) = (packed.id, i as u32);
            let slot = match recent
                .slots
                .iter()
                .position(|slot| slot.0 == id && slot.1 == chunk)
            {
                Some(slot) => slot,
                None => {
                    let slot = if recent.slots.len() < RECENT {
                        recent.slots.push((id, chunk, 0, Box::default()));
                        recent.slots.len() - 1
                    } else {
                        (0..RECENT)
                            .min_by_key(|&s| recent.slots[s].2)
                            .expect("slots")
                    };
                    let entry = &mut recent.slots[slot];
                    (entry.0, entry.1) = (u32::MAX, u32::MAX);
                    if entry.3.len() != length + 1 {
                        entry.3 = vec![0; length + 1].into_boxed_slice();
                    }
                    self.inflate(packed, i, &mut entry.3)?;
                    (entry.0, entry.1) = (id, chunk);
                    slot
                }
            };
            let entry = &mut recent.slots[slot];
            entry.2 = clock;
            let bytes = &entry.3[..length];
            if kept.take(length) && packed.kept[i].set(bytes.into()).is_err() {
                kept.give_back(length);
            }
            Ok(f(bytes))
        })
    }

    /// The CRC-32 of the table's bytes, read as a probe reads them, a chunk
    /// at a time, without keeping any.
    fn crc_of_bytes(
        &self,
        mut each: impl FnMut(&[u8]) -> Result<(), String>,
    ) -> Result<u32, String> {
        let mut crc = Crc::default();
        match &self.kind {
            Kind::Plain => {
                let bytes = &self.map[PLAIN_HEADER..];
                for part in bytes.chunks(1 << 20) {
                    crc.update(part);
                    each(part)?;
                }
            }
            Kind::Packed(packed) => {
                let mut buffer = vec![0; packed.chunk_length + 1];
                for i in 0..packed.kept.len() {
                    self.inflate(packed, i, &mut buffer)?;
                    let out = &buffer[..self.chunk_length(packed, i)];
                    crc.update(out);
                    each(out)?;
                }
            }
        }
        Ok(crc.finish())
    }
}

/// Writes the table `packed` unpacked to `plain`, a new file that the reader
/// takes as it is, checking its bytes against the CRC-32 of gzip's trailer as
/// they are written and again as they read back.
pub fn unpack(packed: &Path, plain: &Path) -> Result<u64, String> {
    let table = TableFile::open(packed)?;
    if table.is_plain() || table.length > 0x7FFF_FFFF {
        return Err(format!(
            "{}: not a dzip file of at most 2 GiB",
            packed.display()
        ));
    }
    let failed = |error: std::io::Error| format!("{}: {error}", plain.display());
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(plain)
        .map_err(failed)?;
    let mut out = BufWriter::with_capacity(1 << 20, file);
    let mut header = [0; PLAIN_HEADER];
    header[..8].copy_from_slice(PLAIN_MAGIC);
    header[8..12].copy_from_slice(&table.mtime.to_le_bytes());
    header[12..16].copy_from_slice(&table.crc.to_le_bytes());
    header[16..24].copy_from_slice(&table.length.to_le_bytes());
    out.write_all(&header).map_err(failed)?;
    let written = table.crc_of_bytes(|bytes| out.write_all(bytes).map_err(failed))?;
    let file = out
        .into_inner()
        .map_err(|error| failed(error.into_error()))?;
    file.sync_all().map_err(failed)?;
    drop(file);
    if written != table.crc {
        return Err(format!(
            "{}: unpacks to bytes of CRC-32 {written:08x}, not its own {:08x}",
            packed.display(),
            table.crc
        ));
    }
    let back = TableFile::open(plain)?;
    let read_back = back.crc_of_bytes(|_| Ok(()))?;
    if !back.is_plain()
        || back.length != table.length
        || back.flags() != table.flags()
        || back.crc != table.crc
        || read_back != table.crc
    {
        return Err(format!(
            "{}: does not read back as {}",
            plain.display(),
            packed.display()
        ));
    }
    Ok(PLAIN_HEADER as u64 + table.length)
}

/// gzip's CRC-32.
struct Crc(u32);

impl Default for Crc {
    fn default() -> Crc {
        Crc(!0)
    }
}

static CRC_TABLE: std::sync::LazyLock<[[u32; 256]; 8]> = std::sync::LazyLock::new(|| {
    let mut table = [[0; 256]; 8];
    for (i, entry) in table[0].iter_mut().enumerate() {
        let mut crc = i as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                0xEDB8_8320 ^ (crc >> 1)
            } else {
                crc >> 1
            };
        }
        *entry = crc;
    }
    for k in 1..8 {
        for i in 0..256 {
            let last = table[k - 1][i];
            table[k][i] = (last >> 8) ^ table[0][(last & 0xFF) as usize];
        }
    }
    table
});

impl Crc {
    /// Eight bytes at a time (slicing by 8).
    fn update(&mut self, bytes: &[u8]) {
        let table = &*CRC_TABLE;
        let mut crc = self.0;
        let (words, rest) = bytes.as_chunks::<8>();
        for word in words {
            let low = crc ^ u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            crc = table[7][(low & 0xFF) as usize]
                ^ table[6][(low >> 8 & 0xFF) as usize]
                ^ table[5][(low >> 16 & 0xFF) as usize]
                ^ table[4][(low >> 24) as usize]
                ^ table[3][word[4] as usize]
                ^ table[2][word[5] as usize]
                ^ table[1][word[6] as usize]
                ^ table[0][word[7] as usize];
        }
        for &byte in rest {
            crc = (crc >> 8) ^ table[0][((crc ^ u32::from(byte)) & 0xFF) as usize];
        }
        self.0 = crc;
    }

    fn finish(&self) -> u32 {
        !self.0
    }
}

#[cfg(test)]
pub mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use miniz_oxide::deflate::core::{
        CompressorOxide, TDEFLFlush, compress, create_comp_flags_from_zip_params,
    };

    use super::{Crc, Kept, PLAIN_HEADER, TableFile, unpack};

    /// A block's bytes unpacked, as in tolius's tables.
    const BLOCK: usize = 58_969;
    /// The flags of his tables of big values.
    const FLAGS: u16 = 7;

    pub fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("provari-tables-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// `data` packed as dictzip packs it: one deflate stream, flushed in
    /// full after each chunk, so that each inflates on its own.
    pub fn pack(data: &[u8], flags: u16) -> Vec<u8> {
        let mut state = CompressorOxide::new(create_comp_flags_from_zip_params(9, -15, 0));
        let chunks: Vec<&[u8]> = data.chunks(BLOCK).collect();
        let mut packed = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let flush = if i + 1 == chunks.len() {
                TDEFLFlush::Finish
            } else {
                TDEFLFlush::Full
            };
            let mut out = vec![0; 2 * BLOCK];
            let (_, read, written) = compress(&mut state, chunk, &mut out, flush);
            assert_eq!(read, chunk.len());
            out.truncate(written);
            packed.push(out);
        }
        let mut bytes = vec![0x1f, 0x8b, 8, 4 | 8];
        bytes.extend((flags as u32).to_le_bytes());
        bytes.extend([2, 3]);
        bytes.extend((10 + 2 * packed.len() as u16).to_le_bytes());
        bytes.extend(b"RA");
        bytes.extend((6 + 2 * packed.len() as u16).to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend((BLOCK as u16).to_le_bytes());
        bytes.extend((packed.len() as u16).to_le_bytes());
        for chunk in &packed {
            bytes.extend((chunk.len() as u16).to_le_bytes());
        }
        bytes.extend(b"table\0");
        for chunk in &packed {
            bytes.extend(chunk);
        }
        let mut crc = Crc::default();
        crc.update(data);
        bytes.extend(crc.finish().to_le_bytes());
        bytes.extend((data.len() as u32).to_le_bytes());
        bytes
    }

    fn read(file: &TableFile, start: usize, size: usize, kept: &Kept) -> Vec<u8> {
        let mut bytes = vec![0; size];
        file.read(start as u64, &mut bytes, kept).unwrap();
        bytes
    }

    /// The packed file and the unpacked one read alike: length, flags and
    /// every byte, block by block and across blocks, kept or not.
    fn assert_read_alike(packed: &Path, plain: &Path, data: &[u8]) {
        let (packed, plain) = (
            TableFile::open(packed).unwrap(),
            TableFile::open(plain).unwrap(),
        );
        assert!(plain.is_plain() && !packed.is_plain());
        assert_eq!(
            (plain.length, plain.flags(), plain.crc),
            (packed.length, packed.flags(), packed.crc)
        );
        let (none, all) = (Kept::default(), Kept::default());
        all.set_limit(usize::MAX);
        for kept in [&none, &all, &all] {
            for start in (0..data.len()).step_by(BLOCK) {
                let size = BLOCK.min(data.len() - start);
                assert_eq!(read(&packed, start, size, kept), &data[start..start + size]);
                assert_eq!(read(&plain, start, size, kept), &data[start..start + size]);
            }
            for start in [BLOCK - 8, 2 * BLOCK - 1, data.len() - 3] {
                assert_eq!(read(&packed, start, 3, kept), &data[start..start + 3]);
                assert_eq!(read(&plain, start, 3, kept), &data[start..start + 3]);
            }
        }
        assert_eq!(none.bytes(), 0);
        assert_eq!(all.bytes(), data.len());
        assert!(
            packed
                .read(data.len() as u64 - 1, &mut [0; 2], &none)
                .is_err()
        );
    }

    #[test]
    fn an_unpacked_table_reads_as_its_packed_file() {
        let dir = scratch("unpacked-table");
        // Over three blocks and a short one, a pattern that packs.
        let data: Vec<u8> = (0..3 * BLOCK + 1000)
            .map(|i| (((i * 7) % 251) ^ (i / 1000)) as u8)
            .collect();
        let packed = dir.join("KvK.an0");
        fs::write(&packed, pack(&data, FLAGS)).unwrap();
        let plain = dir.join("plain.an0");
        let bytes = unpack(&packed, &plain).unwrap();
        assert_eq!(bytes, (PLAIN_HEADER + data.len()) as u64);
        assert_eq!(fs::metadata(&plain).unwrap().len(), bytes);
        assert_read_alike(&packed, &plain, &data);
        assert_eq!(TableFile::open(&plain).unwrap().flags(), FLAGS);

        // Nothing is written over, and an unpacked table is not unpacked
        // again.
        assert!(unpack(&packed, &plain).is_err());
        assert!(unpack(&plain, &dir.join("again.an0")).is_err());
        assert!(!dir.join("again.an0").exists());
        // A packed file whose bytes are not what its CRC says leaves its
        // unpacked file unfinished, and refused.
        let mut wrong = pack(&data, FLAGS);
        let at = wrong.len() - 8;
        wrong[at] ^= 1;
        fs::write(dir.join("wrong.an0"), wrong).unwrap();
        let error = unpack(&dir.join("wrong.an0"), &dir.join("wrong-plain.an0")).unwrap_err();
        assert!(error.contains("CRC-32"), "{error}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_reader_refuses_what_is_neither_a_packed_nor_an_unpacked_table() {
        let dir = scratch("refused");
        let data = vec![3u8; BLOCK + 10];
        let packed = dir.join("packed.an0");
        fs::write(&packed, pack(&data, FLAGS)).unwrap();
        let plain = dir.join("plain.an0");
        unpack(&packed, &plain).unwrap();
        let mut short = fs::read(&plain).unwrap();
        short.pop();
        let mut cut = fs::read(&packed).unwrap();
        cut.truncate(40);
        let cases = [
            (
                "gunzipped",
                b"bytes as gunzip leaves them, flags only in the date".to_vec(),
                "neither",
            ),
            ("short", short, "not the"),
            ("cut", cut, "chunks"),
        ];
        for (case, bytes, refusal) in cases {
            let path = dir.join(format!("{case}.an0"));
            fs::write(&path, bytes).unwrap();
            let error = TableFile::open(&path).err().unwrap_or_default();
            assert!(error.contains(refusal), "{case}: {error}");
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn crc_is_gzips() {
        let mut crc = Crc::default();
        crc.update(b"123456789");
        assert_eq!(crc.finish(), 0xCBF4_3926);
        let text = b"The quick brown fox jumps over the lazy dog, again and again";
        let mut parts = Crc::default();
        parts.update(&text[..13]);
        parts.update(&text[13..]);
        let mut whole = Crc::default();
        whole.update(text);
        assert_eq!(parts.finish(), whole.finish());
    }
}
