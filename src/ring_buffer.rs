use std::collections::VecDeque;

pub struct RingBuffer {
    buffer: VecDeque<f32>,
    capacity: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, data: &[f32]) {
        for &sample in data {
            if self.buffer.len() == self.capacity {
                self.buffer.pop_front();
            }
            self.buffer.push_back(sample);
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Read `count` samples without removing them
    pub fn peek(&self, count: usize) -> Vec<f32> {
        self.buffer.iter().take(count).cloned().collect()
    }

    /// Read `count` samples and remove `advance` samples
    /// This is useful for sliding windows: read window_size, advance hop_size
    pub fn read_and_advance(&mut self, read_count: usize, advance_count: usize) -> Option<Vec<f32>> {
        if self.buffer.len() < read_count {
            return None;
        }

        let data: Vec<f32> = self.buffer.iter().take(read_count).cloned().collect();
        
        // Remove `advance_count` samples
        // If advance_count > read_count, we remove more than we read (skipping)
        // If advance_count < read_count, we have overlap
        let to_remove = advance_count.min(self.buffer.len());
        for _ in 0..to_remove {
            self.buffer.pop_front();
        }

        Some(data)
    }
}
