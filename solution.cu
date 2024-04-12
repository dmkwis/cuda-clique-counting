#include "utils.h"
#include "types.h"
#include "preprocessing.h"
#include "direct_edges.h"
#include "orientation_count.h"
#include <vector>
#include <utility>

int main(int argc, char **argv) {
    /** VALIDATE ARGS **/
    if(argc != 4) {
        error("Usage: " + std::string(argv[0]) +
              " <graph input file> <k value> <output file>");
    }
    std::string input_file_name = std::string(argv[1]);
    int k = atoi(argv[2]);
    std::string output_file_name = std::string(argv[3]);
    if(k == 0) {
        error("Can't convert k value or k is equal to 0 (incorrect value)");
    }
    if(k < 0 || k > 12) {
        error("Incorrect k value, should be a positive integer <= 12");
    }

    /** READ INPUT **/
    std::pair<uint32_t, input_t> input_desc = read_input(input_file_name);
    uint32_t vert_num = input_desc.first;
    uint32_t edge_num = input_desc.second.size();
    input_t input = input_desc.second;

    /** CREATE CSR GRAPH REPRESENTATION **/
    adjecency_list_t al = input_to_adjecency_list(input, vert_num);
    CSR_graph g = create_CSR_graph(al, edge_num);

    /** DIRECT THE EDGES OF G**/
    CSR_graph directed_g = direct_edges(g);
    
    // GRAPH PREPROCESSING FINISHED 
    uint32_t *answer = new uint32_t[K_BOUND];;
    memset(answer, 0, sizeof(uint32_t) * K_BOUND);
    answer[1] = directed_g.vert_num;
    answer[2] = directed_g.edge_num;
    if(k > 2) {
        for(int i = 3; i < K_BOUND; ++i) {
            answer[i] = 0;
        }
        count_k_cliques(directed_g, k, answer);
    }
    
    save_answer(output_file_name, answer, k);
}