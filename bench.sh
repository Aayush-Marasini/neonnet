#!/bin/bash

# 1. Lock the CPU to maximum frequency
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null

echo -e "\n=== 1. Pure Latency Benchmark ==="
taskset -c 3 ./neonnet_bench

echo -e "\n=== 2. Hardware Counters (perf stat) ==="
# sudo is required to read physical hardware counters
sudo perf stat -e cycles,instructions,cache-references,cache-misses taskset -c 3 ./neonnet_bench

echo -e "\n=== 3. Profiling Hotspots (perf record) ==="
# sudo is required to sample the CPU
# -g enables call-graph (so we can see who called who if it gets inlined)
sudo perf record -g -o perf.data -- taskset -c 3 ./neonnet_bench

# 4. Release the CPU governor back to normal
echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null

echo -e "\nProfiling complete. Run 'sudo perf report' to view hotspots."
