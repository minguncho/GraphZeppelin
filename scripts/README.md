# GPUSketch Benchmark Scripts

Note: 
- To ensure the best performance for the GPU, make sure to maximize its clock speed by `nvidia-smi -lgc [CLOCK_SPEED]`.
- Make sure that all the exectuables for tests and tools are getting built, with `cmake -DBUILD_TEST_AND_TOOL=ON ..`.

### Datasets
| Dataset | # of Nodes | # of Edges | # Stream Updates |
| :--- | :--- | :--- | :--- |
| kron13 | $2^{13}$ | $17M$ | $18M$ |
| kron15 | $2^{15}$ | $270M$ | $280M$ |
| kron16 | $2^{16}$ | $1.1B$ | $1.1B$ |
| kron17 | $2^{17}$ | $4.3B$ | $4.5B$ |
| ktree13 | $2^{13}$ | $15M$ | $15M$ |
| ktree15 | $2^{15}$ | $230M$ | $230M$ |
| ktree16 | $2^{16}$ | $940M$ | $940M$ |
| ktree17 | $2^{17}$ | $3.8B$ | $3.8B$ |
| ca_citeseer | $227K$ | $810K$ | $1.6M$ |
| google_plus | $107K$ | $14M$ | $27M$ |
| p2p_gnutella | $63K$ | $150K$ | $294K$ |
| rec_amazon | $92K$ | $130K$ | $250K$ |
| web_uk | $130K$ | $12M$ | $23M$ |
| reddit_corpus | $1M$ | $66B$ | $66B$ |

For evaluating the connected components problem, we only use the kron and the real-world sparse (`ca_citeseer`, `google_plus`, `p2p_gnutella`, `rec_amazon`, `web_uk`) graphs.
For evaluating the minimum cut problem, we use all of the above graphs. Note that we use a modified version of the kron graphs, customized by the addition of edge insertions to control the number of minimum cuts, to evaluate the approximation of the minimum cut values. 

### Benchmark List
- `connected_components.sh`: Connected Components performance benchmark on CPU-GPU system.
- `minimum_cut.sh`: Minimum Cut performance benchmark on CPU-GPU system with hybrid data structure.
- `minimum_cut_no_es.sh`: Minimum Cut performance benchmark on CPU-GPU system with original [Ahn et al.](https://dl.acm.org/doi/10.1145/2213556.2213560) algorithm.
- `minimum_cut_approx.sh`: Minimum Cut approximation benchmark with different values of `epsilon` on `kron_17` and `ktree_17` graphs.
- `sys_sketch_throughput.sh`: Sketch update throughput benchmark on CPU and GPU. 
