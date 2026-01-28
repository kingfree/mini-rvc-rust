pub struct CrossFadeStitcher {
    fade_len: usize,
    last_tail: Vec<f32>,
}

impl CrossFadeStitcher {
    pub fn new(fade_len: usize) -> Self {
        Self {
            fade_len,
            last_tail: Vec::new(),
        }
    }

    /// Process a new chunk of audio.
    /// This assumes the chunks are consecutive in time.
    /// It crossfades the start of the new chunk with the end of the previous chunk.
    pub fn process(&mut self, mut chunk: Vec<f32>) -> Vec<f32> {
        if self.last_tail.is_empty() {
            // First chunk, no stitching needed yet.
            // Save the tail for next time.
            if chunk.len() > self.fade_len {
                let split_idx = chunk.len() - self.fade_len;
                self.last_tail = chunk.split_off(split_idx);
                return chunk;
            } else {
                // Chunk too short, just save it all? 
                // This shouldn't happen in typical streaming where chunks are large enough.
                self.last_tail = chunk;
                return Vec::new();
            }
        }

        // We have a tail from previous chunk.
        // Crossfade it with the beginning of the new chunk.
        let fade_len = self.fade_len.min(chunk.len()).min(self.last_tail.len());
        
        for i in 0..fade_len {
            let alpha = i as f32 / fade_len as f32; // 0.0 -> 1.0
            // tail fades out (1 -> 0), chunk fades in (0 -> 1)
            chunk[i] = self.last_tail[i] * (1.0 - alpha) + chunk[i] * alpha;
        }

        // Save new tail
        if chunk.len() > self.fade_len {
            let split_idx = chunk.len() - self.fade_len;
            self.last_tail = chunk.split_off(split_idx);
        } else {
            // Should not happen if chunks are big enough
            self.last_tail = chunk.clone();
            chunk.clear();
        }

        chunk
    }
    
    pub fn reset(&mut self) {
        self.last_tail.clear();
    }
}
