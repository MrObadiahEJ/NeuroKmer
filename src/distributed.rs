// src/distributed.rs
// Distributed computing support for unlimited scale
// Uses message passing for cluster-wide k-mer counting

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

/// Message types for distributed computation
#[derive(Debug, Clone)]
pub enum DistributedMessage {
    // Worker registration
    RegisterWorker { worker_id: String, address: String },
    
    // Task distribution
    TaskAssignment { task_id: u64, seq_data: Vec<u8> },
    TaskComplete { task_id: u64, results: TaskResults },
    
    // Aggregation
    RequestGlobalCount { kmer: u64 },
    GlobalCountResponse { kmer: u64, count: u64 },
    
    // Control
    Shutdown,
    Heartbeat { worker_id: String, load: f32 },
}

#[derive(Debug, Clone)]
pub struct TaskResults {
    pub kmer_counts: HashMap<u64, u32>,
    pub neuron_currents: Vec<(usize, u64)>, // (neuron_idx, current)
    pub processed_bytes: usize,
}

/// Distributed coordinator (master node)
pub struct DistributedCoordinator {
    pub address: String,
    pub workers: Arc<Mutex<HashMap<String, WorkerInfo>>>,
    pub task_queue: Arc<Mutex<Vec<Task>>>,
    pub results: Arc<Mutex<HashMap<u64, u64>>>, // Global k-mer counts
    pub next_task_id: AtomicU64,
    pub shutdown: Arc<AtomicU64>,
}

#[derive(Debug)]
pub struct WorkerInfo {
    pub address: String,
    pub last_heartbeat: std::time::Instant,
    pub current_load: f32,
    pub total_processed: u64,
}

#[derive(Debug)]
pub struct Task {
    pub id: u64,
    pub data: Vec<u8>,
    pub assigned_to: Option<String>,
}

impl DistributedCoordinator {
    pub fn new(bind_address: &str) -> Self {
        Self {
            address: bind_address.to_string(),
            workers: Arc::new(Mutex::new(HashMap::new())),
            task_queue: Arc::new(Mutex::new(Vec::new())),
            results: Arc::new(Mutex::new(HashMap::new())),
            next_task_id: AtomicU64::new(0),
            shutdown: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Start coordinator server
    pub fn run(&self) -> std::io::Result<()> {
        let listener = TcpListener::bind(&self.address)?;
        println!("Distributed coordinator listening on {}", self.address);
        
        let (tx, rx) = unbounded::<DistributedMessage>();
        
        // Spawn message handler thread
        let workers = self.workers.clone();
        let results = self.results.clone();
        let shutdown = self.shutdown.clone();
        
        thread::spawn(move || {
            Self::handle_messages(rx, workers, results, shutdown);
        });

        // Accept connections
        for stream in listener.incoming() {
            if self.shutdown.load(Ordering::Relaxed) == 1 {
                break;
            }
            
            match stream {
                Ok(stream) => {
                    let tx = tx.clone();
                    thread::spawn(move || {
                        Self::handle_connection(stream, tx);
                    });
                }
                Err(e) => eprintln!("Connection failed: {}", e),
            }
        }
        
        Ok(())
    }

    fn handle_connection(mut stream: TcpStream, tx: Sender<DistributedMessage>) {
        let mut buffer = vec![0u8; 65536];
        
        loop {
            match stream.read(&mut buffer) {
                Ok(0) => break, // Connection closed
                Ok(n) => {
                    // Deserialize message (simplified - use bincode in production)
                    if let Ok(msg) = Self::deserialize_message(&buffer[..n]) {
                        let _ = tx.send(msg);
                    }
                }
                Err(e) => {
                    eprintln!("Read error: {}", e);
                    break;
                }
            }
        }
    }

    fn handle_messages(
        rx: Receiver<DistributedMessage>,
        workers: Arc<Mutex<HashMap<String, WorkerInfo>>>,
        results: Arc<Mutex<HashMap<u64, u64>>>,
        shutdown: Arc<AtomicU64>,
    ) {
        while shutdown.load(Ordering::Relaxed) == 0 {
            match rx.recv_timeout(std::time::Duration::from_secs(1)) {
                Ok(msg) => match msg {
                    DistributedMessage::RegisterWorker { worker_id, address } => {
                        println!("Worker {} registered from {}", worker_id, address);
                        workers.lock().unwrap().insert(worker_id, WorkerInfo {
                            address,
                            last_heartbeat: std::time::Instant::now(),
                            current_load: 0.0,
                            total_processed: 0,
                        });
                    }
                    
                    DistributedMessage::TaskComplete { task_id, results: task_results } => {
                        println!("Task {} completed", task_id);
                        
                        // Merge results
                        let mut global = results.lock().unwrap();
                        for (kmer, count) in task_results.kmer_counts {
                            *global.entry(kmer).or_insert(0) += count as u64;
                        }
                    }
                    
                    DistributedMessage::Heartbeat { worker_id, load } => {
                        if let Some(worker) = workers.lock().unwrap().get_mut(&worker_id) {
                            worker.last_heartbeat = std::time::Instant::now();
                            worker.current_load = load;
                        }
                    }
                    
                    DistributedMessage::Shutdown => {
                        shutdown.store(1, Ordering::Relaxed);
                        break;
                    }
                    
                    _ => {}
                },
                Err(_) => continue, // Timeout
            }
        }
    }

    fn deserialize_message(_data: &[u8]) -> Result<DistributedMessage, ()> {
        // Placeholder - implement proper serialization
        Err(())
    }

    /// Distribute task to available workers
    pub fn distribute_task(&self, seq_data: Vec<u8>) -> u64 {
        let task_id = self.next_task_id.fetch_add(1, Ordering::Relaxed);
        
        let task = Task {
            id: task_id,
            data: seq_data,
            assigned_to: None,
        };
        
        self.task_queue.lock().unwrap().push(task);
        task_id
    }

    /// Get global aggregated count
    pub fn get_global_count(&self, kmer: u64) -> u64 {
        self.results.lock().unwrap().get(&kmer).copied().unwrap_or(0)
    }
}

/// Distributed worker node
pub struct DistributedWorker {
    pub coordinator_address: String,
    pub worker_id: String,
    pub local_counter: crate::spiking_hash::SpikingKmerCounter,
    pub task_sender: Option<Sender<DistributedMessage>>,
}

impl DistributedWorker {
    pub fn new(
        coordinator: &str,
        worker_id: &str,
        k: usize,
        pool_size: usize,
    ) -> Self {
        Self {
            coordinator_address: coordinator.to_string(),
            worker_id: worker_id.to_string(),
            local_counter: crate::spiking_hash::SpikingKmerCounter::new(
                k, 1.0, 0.95, 2, 1.0, pool_size, true
            ),
            task_sender: None,
        }
    }

    /// Connect to coordinator and start processing
    pub fn run(&mut self) -> std::io::Result<()> {
        // Register with coordinator
        let stream = TcpStream::connect(&self.coordinator_address)?;
        let register_msg = DistributedMessage::RegisterWorker {
            worker_id: self.worker_id.clone(),
            address: stream.local_addr()?.to_string(),
        };
        
        self.send_message(&stream, &register_msg)?;
        
        // Start heartbeat thread
        let worker_id = self.worker_id.clone();
        let coordinator = self.coordinator_address.clone();
        
        thread::spawn(move || {
            Self::heartbeat_loop(worker_id, coordinator);
        });

        // Process incoming tasks
        self.process_tasks(stream)
    }

    fn process_tasks(&mut self, mut stream: TcpStream) -> std::io::Result<()> {
        let mut buffer = vec![0u8; 65536];
        
        loop {
            match stream.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => {
                    if let Ok(msg) = Self::deserialize_message(&buffer[..n]) {
                        match msg {
                            DistributedMessage::TaskAssignment { task_id, seq_data } => {
                                println!("Worker {} processing task {}", self.worker_id, task_id);
                                
                                // Process locally
                                let results = self.process_task(&seq_data);
                                
                                // Send results back
                                let complete_msg = DistributedMessage::TaskComplete {
                                    task_id,
                                    results,
                                };
                                self.send_message(&stream, &complete_msg)?;
                            }
                            
                            DistributedMessage::Shutdown => {
                                println!("Worker {} shutting down", self.worker_id);
                                break;
                            }
                            
                            _ => {}
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Worker {} error: {}", self.worker_id, e);
                    break;
                }
            }
        }
        
        Ok(())
    }

    fn process_task(&mut self, seq_data: &[u8]) -> TaskResults {
        // Parse sequences from data
        let seqs = vec![seq_data.to_vec()]; // Simplified
        
        // Process with local counter
        self.local_counter.process_parallel(&seqs);
        
        // Extract results
        let kmer_counts: HashMap<u64, u32> = self.local_counter.counts.iter()
            .map(|entry| (*entry.key(), entry.value().load(Ordering::Relaxed)))
            .collect();

        let neuron_currents: Vec<(usize, u64)> = self.local_counter.neuron_currents.iter()
            .enumerate()
            .map(|(idx, current)| (idx, current.load(Ordering::Relaxed)))
            .filter(|(_, c)| *c > 0)
            .collect();

        TaskResults {
            kmer_counts,
            neuron_currents,
            processed_bytes: seq_data.len(),
        }
    }

    fn heartbeat_loop(worker_id: String, coordinator: String) {
        loop {
            thread::sleep(std::time::Duration::from_secs(30));
            
            // Send heartbeat
            if let Ok(stream) = TcpStream::connect(&coordinator) {
                let msg = DistributedMessage::Heartbeat {
                    worker_id: worker_id.clone(),
                    load: 0.5, // TODO: measure actual load
                };
                let _ = Self::send_message_static(&stream, &msg);
            }
        }
    }

    fn send_message(&self, stream: &TcpStream, msg: &DistributedMessage) -> std::io::Result<()> {
        Self::send_message_static(stream, msg)
    }

    fn send_message_static(mut stream: &TcpStream, _msg: &DistributedMessage) -> std::io::Result<()> {
        // Placeholder - implement proper serialization
        stream.write_all(b"message")?;
        Ok(())
    }

    fn deserialize_message(_data: &[u8]) -> Result<DistributedMessage, ()> {
        Err(())
    }
}

/// High-level distributed API
pub struct DistributedKmerCounter {
    pub mode: DistributedMode,
    pub coordinator: Option<DistributedCoordinator>,
    pub worker: Option<DistributedWorker>,
}

pub enum DistributedMode {
    Standalone,
    Coordinator { address: String },
    Worker { coordinator: String, id: String },
}

impl DistributedKmerCounter {
    pub fn new_coordinator(address: &str) -> Self {
        Self {
            mode: DistributedMode::Coordinator { address: address.to_string() },
            coordinator: Some(DistributedCoordinator::new(address)),
            worker: None,
        }
    }

    pub fn new_worker(coordinator: &str, worker_id: &str, k: usize, pool_size: usize) -> Self {
        Self {
            mode: DistributedMode::Worker {
                coordinator: coordinator.to_string(),
                id: worker_id.to_string(),
            },
            coordinator: None,
            worker: Some(DistributedWorker::new(coordinator, worker_id, k, pool_size)),
        }
    }

    pub fn run(self) -> std::io::Result<()> {
        match self.mode {
            DistributedMode::Coordinator { .. } => {
                if let Some(coord) = self.coordinator {
                    coord.run()?;
                }
            }
            DistributedMode::Worker { .. } => {
                if let Some(mut worker) = self.worker {
                    worker.run()?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Distributed map-reduce for k-mer counting
pub struct MapReduceKmer {
    pub chunks: Vec<Vec<u8>>,
    pub chunk_size: usize,
}

impl MapReduceKmer {
    pub fn new(data: Vec<u8>, chunk_size: usize) -> Self {
        let chunks: Vec<Vec<u8>> = data.chunks(chunk_size).map(|c| c.to_vec()).collect();
        Self { chunks, chunk_size }
    }

    /// Map phase: distribute chunks to workers
    pub fn map<F>(&self, mapper: F) -> Vec<TaskResults>
    where
        F: Fn(&[u8]) -> TaskResults + Send + Sync,
    {
        use rayon::prelude::*;
        
        self.chunks.par_iter()
            .map(|chunk| mapper(chunk))
            .collect()
    }

    /// Reduce phase: aggregate results
    pub fn reduce(results: &[TaskResults]) -> HashMap<u64, u64> {
        let mut global = HashMap::new();
        
        for result in results {
            for (&kmer, &count) in &result.kmer_counts {
                *global.entry(kmer).or_insert(0) += count as u64;
            }
        }
        
        global
    }
}