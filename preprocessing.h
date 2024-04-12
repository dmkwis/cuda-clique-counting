#ifndef PREPROCESSING_H
#define PREPROCESSING_H
#include <vector>
#include "types.h"

CSR_graph create_CSR_graph(const adjecency_list_t &al, uint32_t edge_num) {
    CSR_graph g; 
    g.edge_num = edge_num;
    g.vert_num = (uint32_t)al.size();
    g.V = new uint32_t[g.vert_num];
    g.E = new uint32_t[g.edge_num];

    uint32_t v_idx = 0;
    uint32_t e_idx = 0;

    g.V[0] = 0;
    while(v_idx < al.size()) {
        for(const auto dest: al[v_idx]) {
            g.E[e_idx] = dest;
            ++e_idx;
        }
        ++v_idx;
        if(v_idx == al.size()) break;
        g.V[v_idx] = e_idx;
    }
    return g;
}

void print_CSR_graph(CSR_graph g) {
    std::cout << g.vert_num << " " << g.edge_num << std::endl;
    for(uint32_t i = 0; i < g.vert_num; ++i) {
        std::cout << i << ": ";
        uint32_t beg = g.V[i];
        uint32_t end = (i + 1 == g.vert_num)?(g.edge_num):g.V[i+1];
        for(uint32_t i = beg; i < end; ++i) {
            std::cout << g.E[i] << " ";
        }
        std::cout << std::endl;
    }
}

adjecency_list_t input_to_adjecency_list(const input_t &input, uint32_t vert_num) {
    adjecency_list_t al = std::vector<std::set<uint32_t>>(vert_num, std::set<uint32_t>());
    for(const auto& [from, to]: input) {
        al[from].insert(to);
    }
    return al;
}
#endif //PREPROCESSING_H