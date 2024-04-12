#ifndef TYPES_H
#define TYPES_H
#include <vector>
#include <set>
#include <utility>
using input_t = std::set<std::pair<uint32_t, uint32_t>>;
using adjecency_list_t = std::vector<std::set<uint32_t>>;

struct CSR_graph {
    uint32_t edge_num;
    uint32_t vert_num;
    uint32_t *V;
    uint32_t *E;
};

#endif //TYPES_H