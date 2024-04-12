#ifndef DIRECT_EDGES_H
#define DIRECT_EDGES_H
#include "types.h"
#include "vertex_ordering.h"


namespace edge_directing {
    __device__ inline bool is_before(uint32_t a, uint32_t b, uint32_t *ordering) {
        if(ordering[a] == ordering[b]) {
            return a < b;
        }
        return ordering[a] > ordering[b];
    }
    /*
        Needs one thread per vertex.
    */
    __global__ void calculate_new_out_degrees(uint32_t *V, uint32_t *E,
                                            uint32_t vert_num, uint32_t edge_num,
                                            uint32_t *ordering, uint32_t *new_V) {
        uint32_t vertex_num = blockDim.x * blockIdx.x + threadIdx.x;
        uint32_t remaining_edges = 0;
        if(vertex_num < vert_num) {
            uint32_t beg = V[vertex_num];
            uint32_t end = (vertex_num + 1 == vert_num)?(edge_num):(V[vertex_num+1]);
            for(uint32_t i = beg; i < end; ++i) {
                if(is_before(E[i], vertex_num, ordering)) {
                    remaining_edges += 1;
                }
            }
            new_V[vertex_num] = remaining_edges;
        }
    }
    const uint32_t TPB = 1024;
    void calculate_new_out_degrees_interface(CSR_graph g, uint32_t *ordering, 
                                        uint32_t *new_V) {
        uint32_t *V = g.V;
        uint32_t *E = g.E;
        uint32_t vert_num = g.vert_num;
        uint32_t edge_num = g.edge_num;
        uint32_t *dev_V, *dev_E, *dev_ord, *dev_new_V;
        uint32_t vert_size = sizeof(uint32_t) * vert_num;
        uint32_t edge_size = sizeof(uint32_t) * edge_num;
        cudaMalloc((void**)&dev_V, vert_size); 
        cudaMemcpy(dev_V, V, vert_size, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&dev_E, edge_size);
        cudaMemcpy(dev_E, E, edge_size, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&dev_ord, vert_size);
        cudaMemcpy(dev_ord, ordering, vert_size, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&dev_new_V, vert_size);
        cudaMemset(dev_new_V, 0, vert_size);

        calculate_new_out_degrees<<<(vert_num + TPB - 1)/TPB, TPB>>>
        (dev_V, dev_E, vert_num, edge_num, dev_ord, dev_new_V);

        cudaMemcpy(new_V, dev_new_V, vert_size, cudaMemcpyDeviceToHost);
        cudaFree(dev_V);
        cudaFree(dev_E);
        cudaFree(dev_ord);
        cudaFree(dev_new_V);
    }

    /*
        Currently single block scan.
        Array out should be memset to zero;
    */
    __global__ void exclusive_scan(uint32_t *array, uint32_t arr_size, uint32_t *out) {
        uint32_t tid = threadIdx.x;
        uint32_t bsize = blockDim.x;
        uint32_t *current_buf = array;
        uint32_t *current_out = out;
        uint32_t parity = 0;
        for(uint32_t pot = 1; pot < arr_size; pot *= 2, ++parity) {
            for(uint32_t idx = tid; idx < arr_size; idx += bsize) {
                if(pot <= idx) {
                    int prev_idx = idx - pot;
                    current_out[idx] = current_buf[prev_idx] + current_buf[idx]; //jesli current_out==out to tutaj pairty%2 == 0
                }
                else {
                    current_out[idx] = current_buf[idx];
                }
            }
            if(parity%2 == 0) {
                current_buf = out;
                current_out = array;
            }
            else {
                current_buf = array;
                current_out = out;
            }
            __syncthreads();
            /**
            if(tid == 0) {
                printf("PO %d ITERACJI STAN TABLIC:\n", parity);
                for(uint32_t i = 0; i < arr_size; ++i) {
                   printf("%d ", out[i]);
                }
                printf("\n");
                for(uint32_t i = 0; i < arr_size; ++i) {
                    printf("%d ", array[i]);
                }
                printf("\n");
            }
            
            __syncthreads();
            **/
        }
        if(parity%2 == 1) {
            for(uint32_t idx = tid; idx < arr_size; idx += bsize) {
                array[idx] = out[idx];
            }
        }
        __syncthreads();
        for(uint32_t idx = tid + 1; idx < arr_size; idx += bsize) {
                out[idx] = array[idx - 1];
        }
        if(tid == 0) {
            out[0] = 0;
        }
    }

    void exclusive_scan_interface(uint32_t *array, uint32_t arr_size, uint32_t *out) {
        uint32_t *dev_array, *dev_out;
        uint32_t arr_sz = sizeof(uint32_t) * arr_size;

        cudaMalloc((void**)&dev_array, arr_sz); 
        cudaMemcpy(dev_array, array, arr_sz, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&dev_out, arr_sz);
        cudaMemset(dev_out, 0, arr_sz);

        exclusive_scan<<<1, TPB>>>(dev_array, arr_size, dev_out);

        cudaMemcpy(out, dev_out, arr_sz, cudaMemcpyDeviceToHost);
        cudaFree(dev_array);
        cudaFree(dev_out);
    }

    /*
        Needs one thread per vertex.
    */
    __global__ void calculate_directed_edges(uint32_t *old_V, uint32_t *old_E, 
                                    uint32_t *new_V, uint32_t *ordering,
                                    uint32_t vert_num, uint32_t edge_num,
                                    uint32_t *new_E) {
        uint32_t vertex_num = blockDim.x * blockIdx.x + threadIdx.x;
        uint32_t ptr = new_V[vertex_num];
        if(vertex_num < vert_num) {
            uint32_t beg = old_V[vertex_num];
            uint32_t end = (vertex_num + 1 == vert_num)?(edge_num):(old_V[vertex_num+1]);
            for(uint32_t i = beg; i < end; ++i) {
                if(is_before(old_E[i], vertex_num, ordering)) {
                    new_E[ptr] = old_E[i];
                    ++ptr;
                }
            }
        }
    
    }

    void calculate_directed_edges_interface(CSR_graph old_g, CSR_graph new_g, uint32_t *ordering) { 
        uint32_t vert_num = old_g.vert_num;
        uint32_t edge_num = old_g.edge_num;
        uint32_t *dev_old_V, *dev_old_E, *dev_ord, *dev_new_V, *dev_new_E;
        uint32_t vert_size = sizeof(uint32_t) * old_g.vert_num; //equal to sizeof(uint32_t) * new_g.vert_num
        uint32_t old_edge_size = sizeof(uint32_t) * old_g.edge_num;
        uint32_t new_edge_size = sizeof(uint32_t) * new_g.edge_num;
        cudaMalloc((void**)&dev_old_V, vert_size); 
        cudaMemcpy(dev_old_V, old_g.V, vert_size, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&dev_new_V, vert_size); 
        cudaMemcpy(dev_new_V, new_g.V, vert_size, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&dev_old_E, old_edge_size);
        cudaMemcpy(dev_old_E, old_g.E, old_edge_size, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&dev_ord, vert_size);
        cudaMemcpy(dev_ord, ordering, vert_size, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&dev_new_E, new_edge_size);

        calculate_directed_edges<<<(vert_num + TPB - 1)/TPB, TPB>>>
        (dev_old_V, dev_old_E, dev_new_V, dev_ord, vert_num, edge_num, dev_new_E);

        cudaMemcpy(new_g.E, dev_new_E, new_edge_size, cudaMemcpyDeviceToHost);
        cudaFree(dev_old_V);
        cudaFree(dev_old_E);
        cudaFree(dev_ord);
        cudaFree(dev_new_V);
        cudaFree(dev_new_E);
    }

    CSR_graph direct_edges(CSR_graph g, uint32_t *ordering) {
        CSR_graph directed_g;
        directed_g.vert_num = g.vert_num;
        directed_g.edge_num = g.edge_num/2;
        directed_g.V = new uint32_t[directed_g.vert_num];
        directed_g.E = new uint32_t[directed_g.edge_num];
        calculate_new_out_degrees_interface(g, ordering, directed_g.V);
        /**
        std::cout << "BEFORE EXCLUSIVE SCAN: " << std::endl;
        for(uint32_t i = 0; i < directed_g.vert_num; ++i) {
            std::cout << directed_g.V[i] << " ";
        }
        std::cout << std::endl;
        **/
        exclusive_scan_interface(directed_g.V, directed_g.vert_num, directed_g.V);
        /**
        std::cout << "AFTER EXCLUSIVE SCAN: " << std::endl;
        for(uint32_t i = 0; i < directed_g.vert_num; ++i) {
            std::cout << directed_g.V[i] << " ";
        }
        std::cout << std::endl;
        **/
        calculate_directed_edges_interface(g, directed_g, ordering);

        return directed_g;
    }

}

CSR_graph direct_edges(CSR_graph g) {
    uint32_t *ordering = get_vertex_ordering(g);
    CSR_graph directed_g = edge_directing::direct_edges(g, ordering);
    delete[] ordering;
    return directed_g;
}

#endif //DIRECT_EDGES_H


