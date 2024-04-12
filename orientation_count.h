#ifndef ORIENTATION_COUNT_H
#define ORIENTATION_COUNT_H


namespace orientation_clique_counting {
    #define K_BOUND 13
    #define N_BOUND 1024
    #define MOD (static_cast<uint32_t>(1e9))

    constexpr uint32_t TPB = 128;
    constexpr uint32_t SM = 80;
    constexpr uint32_t TPSM = 2048;
    constexpr uint32_t BN = (SM * TPSM) / TPB;
    constexpr uint32_t N_BOUND_DIV = N_BOUND / 32;

    #define SG_I(A, B, C) ((A) * N_BOUND * N_BOUND_DIV + \
                           (B) * N_BOUND_DIV + \
                           (C))
    
    #define B_I(A, B) ((A) * K_BOUND + \
                       (B))
    
    #define I_I(A, B, C) ((A) * K_BOUND * N_BOUND_DIV + \
                          (B) * N_BOUND_DIV + \
                          (C))

    __constant__ uint32_t max_k;
    __constant__ uint32_t block_num;

    __device__ uint32_t find_idx(uint32_t arr[], uint32_t arr_size, uint32_t val) {
        for(uint32_t i = 0; i < arr_size; ++i) {
            if(arr[i] == val) {
                return i;
            }
        }
        return N_BOUND;
    }

    __global__ void calculate_k_cliques(uint32_t *V, uint32_t *E, 
                                        uint32_t vert_num, uint32_t edge_num, 
                                        uint32_t *next_vertex, uint32_t *subgraph,
                                        uint32_t *I, uint32_t *answer) {
        __shared__ uint32_t local_answer[K_BOUND];
        __shared__ uint32_t local_n_tmp[N_BOUND];
        __shared__ uint32_t current_vertex;
        __shared__ uint32_t sg_size;
        const uint32_t tid = threadIdx.x;
        const uint32_t b_size = blockDim.x;
        const uint32_t b_id = blockIdx.x;
        for(uint32_t i = tid; i < K_BOUND; i += b_size) {
            local_answer[i] = 0;
        }
        __syncthreads();
        for(uint32_t cur_v = b_id; cur_v < vert_num; cur_v += block_num) {
            // POLICZ SUBGRAPH ORAZ I
            uint32_t beg_incl = V[cur_v];
            uint32_t end_excl = (cur_v + 1 == vert_num)?(edge_num):(V[cur_v + 1]);
            uint32_t big_size = end_excl - beg_incl;
            if(big_size > 0) { //TODO: +- 1 check
                // MAMY NUMERY SASIADOW W local_n_tmp
                for(uint32_t idx = beg_incl + tid; idx < end_excl; idx += b_size) {
                    local_n_tmp[idx - beg_incl] = E[idx];
                }
                for(uint32_t idx = tid; idx < N_BOUND; idx += b_size) {
                    for(uint32_t j = 0; j < N_BOUND_DIV; ++j) {
                        subgraph[SG_I(b_id, idx, j)] = 0;
                        I[I_I(b_id, 1, j)] = 0;
                    }
                }
                __syncthreads();
                for(uint32_t idx = tid; idx < big_size; idx += b_size) {
                    uint32_t real_name = local_n_tmp[idx];
                    beg_incl = V[real_name];
                    end_excl = (real_name + 1 == vert_num)?(edge_num):(V[real_name + 1]);
                    for(uint32_t n_idx = beg_incl; n_idx < end_excl; ++n_idx) {
                        uint32_t found_idx = find_idx(local_n_tmp, big_size, E[n_idx]);
                        if(found_idx != N_BOUND) {
                            atomicOr(&subgraph[SG_I(b_id, idx, found_idx/32)], (1<<(found_idx&31)));
                        }
                    }
                    //set I
                    atomicOr(&I[I_I(b_id, 1, idx/32)], (1<<(idx&31)));
                }
                if(tid == 0) {
                    next_vertex[B_I(b_id, 1)] = 0;
                }
                __syncthreads();
                uint32_t k = 2;
                while(k > 1) {
                    if(tid == 0) {
                        current_vertex = next_vertex[B_I(b_id, k-1)];
                    }
                    __syncthreads();
                    if(current_vertex >= big_size) {
                        k--;
                    }
                    else {
                        __syncthreads();
                        if(tid == 0) {
                            while(current_vertex < big_size) {
                                if(I[I_I(b_id, k-1, current_vertex/32)] & (1<<(current_vertex&31))) {
                                    break;
                                }
                                ++current_vertex;
                            }
                        }
                        __syncthreads();
                        //CHECK IF NO VERTEX WAS FOUND
                        if(current_vertex >= big_size) {
                            --k;
                        }
                        else {
                            // INCREMENT LAST VERTEX
                            if(tid == 0) {
                                next_vertex[B_I(b_id, k-1)] = current_vertex + 1;
                            }
                            // OBLICZ I'[k] na podstawie I[k-1] oraz I' size oraz ustaw licznik na last vertex
                            for(uint32_t idx = tid; idx < N_BOUND_DIV; idx += b_size) {
                                I[I_I(b_id, k, idx)] = subgraph[SG_I(b_id, current_vertex, idx)] &
                                                        I[I_I(b_id, k-1, idx)];
                            }
                            __syncthreads();
                            // DELETE CURRENT_VERTEX FROM GRAPH AND COUNT SG_SIZE
                            if(tid == 0) {
                                sg_size = 0;
                                for(uint32_t i = 0; i < N_BOUND_DIV; ++i) {
                                    sg_size += __popc(I[I_I(b_id, k, i)]);
                                }
                                if(k + 1 <= max_k && sg_size > 0) {
                                    local_answer[k+1] += sg_size;
                                    local_answer[k+1] %= MOD;
                                }
                                next_vertex[B_I(b_id, k)] = 0;
                            }
                            __syncthreads();
                            // JESLI TRZEBA SIE DALEJ STOSOWAC TO ODLOZ I NA STOS
                            if(sg_size > 0 && k+1 < max_k) {
                                ++k;
                            }
                        }
                    }
                    __syncthreads();
                }
            }
            __syncthreads();
        }
        if(tid == 0) {
            for(uint32_t idx = 3; idx <= max_k; ++idx) {
                answer[B_I(b_id, idx)] += local_answer[idx];
                answer[B_I(b_id, idx)] %= MOD;
            }
        }
    }

    void calculate_k_cliques_interface(CSR_graph directed_g, const uint32_t k, uint32_t *answer) {
       uint32_t *dev_V, *dev_E, *dev_next_vertex, *dev_answer, *dev_subgraph, *dev_I;

       uint32_t v_size = directed_g.vert_num * sizeof(uint32_t);
       cudaMalloc((void**)&dev_V, v_size);
       cudaMemcpy(dev_V, directed_g.V, v_size, cudaMemcpyHostToDevice);

       uint32_t e_size = directed_g.edge_num * sizeof(uint32_t);
       cudaMalloc((void**)&dev_E, e_size);
       cudaMemcpy(dev_E, directed_g.E, e_size, cudaMemcpyHostToDevice);

       uint32_t answer_size = BN * K_BOUND * sizeof(uint32_t);
       cudaMalloc((void**)&dev_answer, answer_size);
       cudaMemset(dev_answer, 0, answer_size);

       uint32_t next_vertex_size = BN * K_BOUND * sizeof(uint32_t);
       cudaMalloc((void**)&dev_next_vertex, next_vertex_size);
       cudaMemset(dev_next_vertex, 0, next_vertex_size);

       uint32_t subgraph_size = BN * N_BOUND * N_BOUND_DIV * sizeof(uint32_t);
       cudaMalloc((void**)&dev_subgraph, subgraph_size);
       cudaMemset(dev_subgraph, 0, subgraph_size);
       
       uint32_t dev_I_size = BN * K_BOUND * N_BOUND_DIV * sizeof(uint32_t);
       cudaMalloc((void**)&dev_I, dev_I_size);
       cudaMemset(dev_I, 0, dev_I_size);

       cudaMemcpyToSymbol(max_k, &k, sizeof(uint32_t));
       cudaMemcpyToSymbol(block_num, &BN, sizeof(uint32_t));
       
       uint32_t* answer_cpy = new uint32_t[BN * K_BOUND];
       if(answer_cpy == NULL) {
            error("Couldn't allocate array for answer");
       }
       calculate_k_cliques<<<BN, TPB>>>(dev_V, dev_E, directed_g.vert_num, directed_g.edge_num,
                                        dev_next_vertex, dev_subgraph, dev_I, dev_answer);

       cudaMemcpy(answer_cpy, dev_answer, answer_size, cudaMemcpyDeviceToHost);
       cudaFree(dev_V);
       cudaFree(dev_E);
       cudaFree(dev_next_vertex);
       cudaFree(dev_subgraph);
       cudaFree(dev_I);
       cudaFree(dev_answer);

       for(uint32_t b_idx = 0; b_idx < BN; ++b_idx) {
            for(uint32_t current_k = 3; current_k <= k; ++current_k) {
                answer[current_k] += answer_cpy[B_I(b_idx, current_k)];
                answer[current_k] %= MOD;
            }
       }
       
       delete[] answer_cpy;   
    }
}

void count_k_cliques(CSR_graph directed_g, const uint32_t k, uint32_t *answer) {
    orientation_clique_counting::calculate_k_cliques_interface(directed_g, k, answer);
}

#endif //ORIENTATION_COUNT_H