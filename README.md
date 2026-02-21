# 2026_SPAC_Examples_DistributedComputing
Basic demos around parts of the distributed computing pipelines

Branching Threshold Model (CPU)

File: cpu_branching.py

Implements a Galton–Watson branching process:
Extinction when 
𝑅
<
1
R<1

Cascade growth when 
𝑅
>
1
R>1

Phase transition behavior

Run
pip install numpy matplotlib
python cpu_branching.py


Parallel Branching (Multiprocessing)

File: parallel_branching.py

Same model as above, but parallelized using:

concurrent.futures.ProcessPoolExecutor

Demonstrates:

Embarrassingly parallel workloads

CPU-based distributed simulation

Run
python parallel_branching.py


GPU Branching (CuPy / CUDA)

File: gpu_branching.py

Accelerates branching simulation using GPU vectorization.


Thousands to millions of trials are simulated simultaneously.

Requirements

NVIDIA GPU

CUDA installed

Matching CuPy version

Install example:

pip install cupy-cuda12x

Run:

python gpu_branching.py


GPU Mixed Emitters Model

File: gpu_mixed_emitters.py

Extends the cascade model to include:

Single emitters (release 1)

Burst emitters (release many)

Optional super-burst events

This creates an emergent threshold based on:

Burst probability

Burst intensity

Failure probability

This mirrors real distributed propagation systems where some events amplify dramatically.

Run
python gpu_mixed_emitters.py