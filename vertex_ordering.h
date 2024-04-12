/*
    Vertex ordering functions take undirected graph in CSR representation and 
    return a dynamically allocated array of size CSR_graph.vert_num with integers
*/

#ifndef VERTEX_ORDERING_H
#define VERTEX_ORDERING_H
#include "types.h"

namespace hist {
    constexpr uint32_t LOCAL_HIST_SIZE = 4096;
    constexpr uint32_t ARRAY_FRAGMENT_SIZE = 1024;
    constexpr uint32_t BLOCK_SIZE = 1024;
    /*
        Each block calculates occurences of 
        [(blockIdx.x) * LOCAL_HIST_SIZE, (blockIdx.x + 1) * LOCAL_HIST_SIZE - 1]
        on fragment [blockIdx.y * ARRAY_FRAGMENT_SIZE, (blockIdx.y + 1) * ARRAY_FRAGMENT_SIZE - 1]
    */
    __global__ void histogram_kernel(uint32_t *array, uint32_t a_size, uint32_t *bins) {
        __shared__ uint32_t local_hist[LOCAL_HIST_SIZE];
        const uint32_t tid = threadIdx.x;
        const uint32_t v_min = blockIdx.x * LOCAL_HIST_SIZE;
        const uint32_t v_max = (blockIdx.x + 1) * LOCAL_HIST_SIZE - 1;
        const uint32_t idx_min = blockIdx.y * ARRAY_FRAGMENT_SIZE;
        const uint32_t idx_max = (blockIdx.y) * ARRAY_FRAGMENT_SIZE - 1;
        //INITIALIZE local_hist to 0
        for(uint32_t idx = tid; idx < LOCAL_HIST_SIZE; idx += BLOCK_SIZE) {
            local_hist[idx] = 0;
        }
        __syncthreads();
        for(uint32_t idx = idx_min + tid; idx <= idx_max && idx < a_size; idx += BLOCK_SIZE) {
            if(array[idx] >= v_min && array[idx] <= v_max) {
                atomicAdd(&local_hist[array[idx] % LOCAL_HIST_SIZE], 1);
            }
        }
        __syncthreads();
        for(uint32_t idx = tid; idx < LOCAL_HIST_SIZE; idx += BLOCK_SIZE) {
            atomicAdd(&(bins[idx + v_min]), local_hist[idx]);
        }
    }
   /*
        Values in array are from range [0, val_range)
   */
    uint32_t *call_histogram(uint32_t *array, uint32_t a_size, uint32_t val_range) {
        uint32_t *bins = new uint32_t[val_range];

        uint32_t *dev_array, *dev_bins;
        uint32_t bins_bytes = sizeof(uint32_t) * val_range;
        uint32_t arr_bytes = sizeof(uint32_t) * a_size;

        cudaMalloc((void**)&dev_bins, bins_bytes);
        cudaMemset(dev_bins, 0, bins_bytes);
        cudaMalloc((void**)&dev_array, arr_bytes);
        cudaMemcpy(dev_array, array, arr_bytes, cudaMemcpyHostToDevice);

        uint32_t xgd = (val_range + LOCAL_HIST_SIZE - 1)/LOCAL_HIST_SIZE;
        uint32_t ygd = (a_size + ARRAY_FRAGMENT_SIZE - 1)/ARRAY_FRAGMENT_SIZE;
        dim3 grid_dims(xgd, ygd,1);
        histogram_kernel<<<grid_dims, BLOCK_SIZE>>>(dev_array, 
                                                    a_size,
                                                    dev_bins);

        cudaMemcpy(bins, dev_bins, bins_bytes, cudaMemcpyDeviceToHost);

        cudaFree(dev_bins);
        cudaFree(dev_array);
        return bins;
    }
}

uint32_t* get_vertex_ordering(CSR_graph g) {
    uint32_t *bins = hist::call_histogram(g.E, g.edge_num, g.vert_num);
    return bins;
}
#endif //DIRECT_EDGES_H