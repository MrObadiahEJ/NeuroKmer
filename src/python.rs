// src/python.rs - Add PyO3 bindings

use neurokmer::{SpikingKmerCounter, stream_sequences};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
struct PySpikingCounter {
    inner: SpikingKmerCounter,
}

#[pymethods]
impl PySpikingCounter {
    #[new]
    fn new(k: usize, pool_size: usize) -> Self {
        Self {
            inner: SpikingKmerCounter::new(k, 1.0, 0.95, 2, 1.0, pool_size),
        }
    }

    fn process_file(&mut self, path: &str) -> PyResult<()> {
        for seq in stream_sequences(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?
        {
            self.inner.process_sequence(&seq);
        }
        Ok(())
    }

    fn get_counts(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for entry in self.inner.counts.iter() {
            dict.set_item(entry.key().to_string(), *entry.value())?;
        }
        Ok(dict.into())
    }

    fn energy_used(&self) -> f64 {
        self.inner.energy_used()
    }
}

#[pymodule]
fn neurokmer_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySpikingCounter>()?;
    m.add_function(wrap_pyfunction!(pack_kmer_py, m)?)?;
    Ok(())
}

#[pyfunction]
fn pack_kmer_py(kmer: &[u8]) -> u64 {
    neurokmer::pack_kmer(kmer)
}
