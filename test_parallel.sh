#!/bin/bash
# Test Candle with parallel optimization

echo "=== Testing Candle with Parallel Optimization ==="
echo ""

# Get CPU count
CPU_COUNT=$(sysctl -n hw.ncpu)
echo "CPU cores available: $CPU_COUNT"
echo ""

# Test with different thread counts
for THREADS in 1 4 8 $CPU_COUNT; do
    echo "--- Testing with $THREADS threads ---"
    export RAYON_NUM_THREADS=$THREADS

    # Run test and capture timing
    cargo run --release --bin test_full 2>&1 | grep -E "Testing|Timing.*RVC inference|TOTAL"
    echo ""
done
