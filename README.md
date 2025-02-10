# CUDAKernels
This document serves as a log of the progress and knowledge I gained while working through the **100-Day CUDA Challenge** posted in [Umar Jamil's Discord server](https://discord.gg/4Tg4TkJQzE) and studying the **PMPP** (*Programming Massively Parallel Processors*) book. The challenge is inspired by resources from Umar's [GitHub](https://github.com/hkproj/).  

## 📚 Learning Resources
- [Umar Jamil's Discord server](https://discord.gg/4Tg4TkJQzE)
- [PMPP book](https://www.amazon.com/Programming-Massively-Parallel-Processors-CUDA/dp/1492043583)
- [Umar's GitHub](https://github.com/hkproj/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch CUDA Docs](https://pytorch.org/docs/stable/cuda.html)

## 🚀 Overview  
A 100-day challenge to learn CUDA kernel programming, focusing on optimizing LLMs and GraphML models.  
**Goal**: Write a CUDA kernel daily, compare performance against CPU/naive implementations, and document learnings.  

## 📂 Repository Structure
```text
CUDAKernels/
├── Day01_VectorAdd/
│   ├── vector_add.cu       # CUDA kernel code
│   ├── README.md           # Explanation + results
│   └── results.png         # Performance graphs (optional)
├── Day02_MatrixTranspose/
│   └── ...
└── ...
```

## 🚀 Daily Tasks
1. **Code**: Implement a kernel in Google Colab (`.cu` file).
2. **Profile**: Use `nvprof` or `%%timeit` to measure performance.
3. **Compare**: Benchmark against CPU/naive GPU/library implementations.
4. **Document**: Update `README.md` with results and insights.

## 🔍 Comparisons
- **Time**: GPU vs CPU runtime.
- **FLOPS**: Theoretical vs achieved (use `nvprof`).
- **Bandwidth**: Memory throughput (GB/s).
- **Occupancy**: Analyze with `Nsight Compute`.

## 🛠️ Tools
- **Google Colab**: Free GPU access (T4/P100).
- **CUDA Toolkit**: `!apt install nvidia-cuda-toolkit`.
- **Profiling**: `nvprof`, `Nsight Compute`.


## 📅 100-Day CUDA Kernel Challenge Roadmap

### **Days 1–10: CUDA Fundamentals**
| Day | Topic                                  | Chapter | Key Comparison                          | Status |
|-----|----------------------------------------|---------|-----------------------------------------|--------|
| 1   | Vector Addition                        | 2       | CPU (NumPy) vs GPU                      | 🔄     |
| 2   | Matrix Transpose (Coalesced Access)    | 3       | Coalesced vs non-coalesced              | 🔄     |
| 3   | Image Blur (Stencil Operation)         | 3       | Global vs shared memory                 | 🔄     |
| 4   | Matrix Multiplication (Naive)          | 3       | CUDA vs NumPy                           | 🔄     |
| 5   | Tiled Matrix Multiplication            | 5       | Naive vs tiled FLOPS                    | 🔄     |
| 6   | Memory Coalescing Optimization         | 6       | Bandwidth improvement                   | 🔄     |
| 7   | Warp Divergence                        | 4       | Divergent vs uniform warps              | 🔄     |
| 8   | Histogram with Atomics                 | 9       | Atomic vs CPU                           | 🔄     |
| 9   | Parallel Reduction (Sum)               | 10      | Tree reduction vs CPU                   | 🔄     |
| 10  | Prefix Sum (Scan)                      | 11      | Kogge-Stone vs Thrust                   | 🔄     |

### **Days 11–20: Parallel Patterns**
| Day | Topic                                  | Chapter | Key Comparison                          | Status |
|-----|----------------------------------------|---------|-----------------------------------------|--------|
| 11  | Convolution (Constant Memory)          | 7       | Shared vs constant memory               | 🔄     |
| 12  | 5-Point Stencil (Edge Detection)       | 8       | Register tiling vs global               | 🔄     |
| 13  | Merge Sorted Arrays                    | 12      | CUDA vs `std::merge`                    | 🔄     |
| 14  | Sparse Matrix-Vector (COO)             | 14      | CUDA vs SciPy                           | 🔄     |
| 15  | BFS (Vertex-Centric)                   | 15      | CPU vs GPU traversal                    | 🔄     |
| 16  | Radix Sort                             | 13      | CUDA vs `std::sort`                     | 🔄     |
| 17  | GEMM with cuBLAS                       | 16      | Custom kernel vs cuBLAS                 | 🔄     |
| 18  | Softmax                                | -       | Fused vs unfused                        | 🔄     |
| 19  | Layer Normalization                    | -       | CUDA vs PyTorch                         | 🔄     |
| 20  | Attention Score (QKᵀ)                  | 16      | Tiled vs naive                          | 🔄     |

### **Days 21–30: LLM-Oriented Kernels**
| Day | Topic                                  | Chapter | Key Comparison                          | Status |
|-----|----------------------------------------|---------|-----------------------------------------|--------|
| 21  | Multi-Head Attention (Batched GEMM)    | 16      | Looped vs batched cuBLAS                | 🔄     |
| 22  | Flash Attention (Tiling)               | 6       | Memory efficiency                       | 🔄     |
| 23  | Sparse Attention Mask                  | 14      | Dense vs sparse                         | 🔄     |
| 24  | FP16 Matrix Multiplication             | 16      | FP32 vs FP16 throughput                 | 🔄     |
| 25  | Kernel Fusion (Softmax + Dropout)      | 6       | Fused vs separate                       | 🔄     |
| 26  | Rotary Positional Embedding            | -       | CUDA vs PyTorch                         | 🔄     |
| 27  | GELU Activation                        | -       | CUDA vs PyTorch                         | 🔄     |
| 28  | Token Pruning (Masked Reduction)       | 10      | Pruned vs full inference                | 🔄     |
| 29  | LayerNorm Gradient                     | -       | Custom vs PyTorch autograd              | 🔄     |
| 30  | Transformer Block (Fused MHA + FFN)    | 16      | End-to-end latency                      | 🔄     |

### **Days 31–50: GraphML & Advanced Topics**
| Day | Topic                                  | Chapter | Key Comparison                          | Status |
|-----|----------------------------------------|---------|-----------------------------------------|--------|
| 31  | Graph Convolution (Aggregation)        | 15      | Shared vs global memory                 | 🔄     |
| 32  | cuSPARSE SpMV (CSR Format)             | 14      | Custom vs cuSPARSE                      | 🔄     |
| 33  | BFS with Frontiers                     | 15      | Vertex-centric vs edge-centric          | 🔄     |
| 34  | Dynamic Parallelism (Recursive Merge)  | 21      | CPU recursion vs GPU                    | 🔄     |
| 35  | CUDA Streams (Async Execution)         | 20      | Synchronous vs async                    | 🔄     |
| 36  | Atomic-Free Histogram                  | 9       | Privatization vs atomics                | 🔄     |
| 37  | FP16 GEMM with Tensor Cores            | 16      | FP16 vs FP32 FLOPS                      | 🔄     |
| 38  | cuDNN Convolution                      | 16      | Custom vs cuDNN                         | 🔄     |
| 39  | Tiled Softmax                          | 6       | Shared memory vs global                 | 🔄     |
| 40  | Distributed Reduction (Multi-GPU)      | 20      | Single-GPU vs multi-GPU                 | 🔄     |
| 41  | Kernel Autotuning (Iterative MRI)      | 17      | Auto-tuned vs heuristic                 | 🔄     |
| 42  | Graph Sampling (Neighborhood)          | 15      | Shared memory vs global                 | 🔄     |
| 43  | MoE (Mixture of Experts)               | 10      | Atomic vs privatized routing            | 🔄     |
| 44  | cuFFT (Fast Fourier Transform)         | -       | CUDA vs NumPy FFT                       | 🔄     |
| 45  | Memory Usage Analysis (Nsight Compute) | 22      | Occupancy vs performance                | 🔄     |
| 46  | Wavefront BFS                          | 15      | Frontier optimization                   | 🔄     |
| 47  | Warp Shuffle Reduction                 | 4       | Shuffle vs atomic reduction             | 🔄     |
| 48  | Multi-GPU GEMM                         | 20      | Single-GPU vs multi-GPU                 | 🔄     |
| 49  | Mixed Precision Training               | 22      | FP16/FP32 stability                     | 🔄     |
| 50  | LLM Inference Kernel                   | 23      | Custom vs PyTorch                       | 🔄     |

### Key:
- ✅ = **Completed**  
- 🔄 = **In Progress**  

**Contributing**: Suggestions welcome! Open an issue or PR.  
**License**: MIT