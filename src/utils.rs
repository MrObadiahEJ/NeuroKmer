// src/utils.rs
// Borrowed streaming + error handling from needletail examples + neural-rs iterator patterns

use crate::{NeuroResult, RollingKmerHash};
use log::{debug, warn};
use needletail::parse_fastx_file;

/// Stream sequences from FASTA/FASTQ – memory efficient, robust to malformed files
pub fn stream_sequences(path: &str) -> NeuroResult<impl Iterator<Item = Vec<u8>>> {
    let mut reader = parse_fastx_file(path)?;
    let sequences = std::iter::from_fn(move || match reader.next() {
        Some(Ok(record)) => {
            let seq = record.seq().to_vec();
            debug!("Read sequence of length {}", seq.len());
            Some(seq)
        }
        Some(Err(e)) => {
            warn!("Skipping malformed record: {}", e);
            None
        }
        None => None,
    });
    Ok(sequences)
}

pub fn pack_kmer(kmer: &[u8]) -> u64 {
    let mut packed = 0u64;
    for &base in kmer {
        let bits = match base {
            b'A' | b'a' => 0u64,
            b'C' | b'c' => 1u64,
            b'G' | b'g' => 2u64,
            b'T' | b't' => 3u64,
            _ => continue, // Skip N/ambiguous
        };
        packed = (packed << 2) | bits;
    }
    packed
}

/// Generate canonical k-mers (optional reverse complement later)
/// Generate k-mers as slices (no cloning – zero-copy for efficiency)
pub fn generate_kmers<'a>(seq: &'a [u8], k: usize) -> impl Iterator<Item = &'a [u8]> + 'a {
    seq.windows(k)
}

/// Return the canonical (lexicographically smaller) representation of a k-mer
/// and its reverse complement
pub fn canonical_kmer(kmer: &[u8]) -> (u64, bool) {
    let forward = pack_kmer(kmer);
    
    // Generate reverse complement
    let mut reverse = 0u64;
    for &base in kmer.iter().rev() {
        let comp_bits = match base {
            b'A' | b'a' => 3u64,
            b'C' | b'c' => 2u64,
            b'G' | b'g' => 1u64,
            b'T' | b't' => 0u64,
            _ => {
                // If there's an N, just return forward
                return (forward, false);
            }
        };
        reverse = (reverse << 2) | comp_bits;
    }
    
    if forward <= reverse {
        (forward, false)
    } else {
        (reverse, true)
    }
}

/// Pack a k-mer with canonicalization built-in
/// Returns the canonical representation (lexicographically smaller of kmer and its reverse complement)
pub fn pack_canonical(kmer: &[u8]) -> u64 {
    let forward = pack_kmer(kmer);
    
    // Generate reverse complement and pack it
    let mut reverse = 0u64;
    for &base in kmer.iter().rev() {
        let comp_bits = match base {
            b'A' | b'a' => 3u64,  // A -> T (3)
            b'C' | b'c' => 2u64,  // C -> G (2)
            b'G' | b'g' => 1u64,  // G -> C (1)
            b'T' | b't' => 0u64,  // T -> A (0)
            _ => {
                // For N, we can't determine canonical - return forward as fallback
                return forward;
            }
        };
        reverse = (reverse << 2) | comp_bits;
    }
    
    // For k-mers shorter than 32, we need to align the bits
    // The reverse complement should be the same length, so no alignment needed
    // But we need to ensure we're comparing correctly
    std::cmp::min(forward, reverse)
}


/// Process a sequence using rolling hash and canonical k-mers
/// Returns iterator of (canonical_kmer, was_reversed) pairs
pub fn process_sequence_canonical(seq: &[u8], k: usize) -> impl Iterator<Item = (u64, bool)> + '_ {
    let mut rolling = RollingKmerHash::new(k);
    let mut i = 0;

    std::iter::from_fn(move || {
        while i + k <= seq.len() {
            if i == 0 {
                // Initialize rolling hash
                rolling.init(&seq[0..k]);
            } else {
                // Slide window
                let prev_base = seq[i - 1];
                let next_base = seq[i + k - 1];
                rolling.slide(next_base, prev_base);
            }

            let (canonical, was_reversed) = if rolling.forward() <= rolling.reverse_complement() {
                (rolling.forward(), false)
            } else {
                (rolling.reverse_complement(), true)
            };

            i += 1;
            return Some((canonical, was_reversed));
        }
        None
    })
}