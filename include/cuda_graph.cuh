#pragma once
#include <vector>
#include <map>
#include <util.h>
#include <atomic>
#include <cuda_kernel.cuh>

class CudaGraph {
    public: 
        CudaUpdateParams *cudaUpdateParams;
        CudaSketch* cudaSketches;
        long* sketchSeeds;

        std::vector<std::mutex> mutexes;
        //std::atomic<vec_t> offset;
        std::vector<cudaStream_t> streams;
        std::vector<int> offset;

        CudaKernel cudaKernel;

        // Number of threads
        int num_device_threads;
        
        // Number of blocks
        int num_device_blocks;

        int num_host_threads;
        int batch_size;

        bool isInit = false;

        // Default constructor
        CudaGraph() {}

        void configure(CudaUpdateParams* _cudaUpdateParams, CudaSketch* _cudaSketches, long* _sketchSeeds, int _num_host_threads) {
            cudaUpdateParams = _cudaUpdateParams;
            cudaSketches = _cudaSketches;
            sketchSeeds = _sketchSeeds;

            mutexes = std::vector<std::mutex>(cudaUpdateParams[0].num_nodes);

            num_device_threads = 1024;
            num_device_blocks = 1;
            num_host_threads = _num_host_threads;
            for (int i = 0; i < num_host_threads; i++) {
                cudaStream_t stream;
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
                streams.push_back(stream);
                cudaStreamAttachMemAsync(streams[i], &cudaUpdateParams[0].edgeUpdates[i * cudaUpdateParams[0].batch_size]);
            }

            isInit = true;
        };

        void batch_update(int id, node_id_t src, const std::vector<node_id_t> &edges) {
            if (!isInit) {
                std::cout << "CudaGraph has not been initialized!\n";
            }
            // Add first to prevent data conflicts
            //vec_t prev_offset = std::atomic_fetch_add(&offset, edges.size());
            while(cudaUpdateParams[0].edgeWriteEnabled[id] == 0) {}
            cudaUpdateParams[0].edgeWriteEnabled[id] = 0;
            int count = 0;
            vec_t offset = id * cudaUpdateParams[0].batch_size;
            for (vec_t i = offset; i < offset + edges.size(); i++) {
                cudaUpdateParams[0].edgeUpdates[i] = static_cast<vec_t>(concat_pairing_fn(src, edges[count]));
                count++;
            }

            cudaKernel.gtsStreamUpdate(num_device_threads, num_device_blocks, id, src, streams[id], offset, edges.size(), cudaUpdateParams, cudaSketches, sketchSeeds);;
        };
};