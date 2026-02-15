// src/utils.rs
// Borrowed streaming + error handling from needletail examples + neural-rs iterator patterns

use needletail::parse_fastx_file;
use crate::NeuroResult;
use log::{debug, warn};

/// Stream sequences from FASTA/FASTQ – memory efficient, robust to malformed files
pub fn stream_sequences(path: &str) -> NeuroResult<impl Iterator<Item = Vec<u8>>> {
    let mut reader = parse_fastx_file(path)?;
    let sequences = std::iter::from_fn(move || {
        match reader.next() {
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
        }
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
            _ => continue,  // Skip N/ambiguous
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