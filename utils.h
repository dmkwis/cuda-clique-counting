#ifndef UTILS_H
#define UTILS_H
#include <fstream>
#include <unordered_map>
#include <iostream>
#include "types.h"

#define MAX_VERT_NUM 5'000'000
#define MAX_EDGE_NUM 50'000'000


void error(const std::string& msg) {
    std::cout << msg << std::endl;
    exit(1);
}

std::pair<uint32_t, input_t> read_input(const std::string &input_file_name) {
    std::fstream input_file;
    input_file.open(input_file_name, std::fstream::in);
    if(!input_file) {
        error("Couldn't open: " + input_file_name);
    }
    uint32_t from, to;
    std::unordered_map<uint32_t, uint32_t> reorder;
    uint32_t id = 0;
    input_t input;
    while(input_file >> from >> to) {
		if(reorder.emplace(to, id).second) ++id;
		if(reorder.emplace(from, id).second) ++id;
        input.insert({reorder[from], reorder[to]});
        input.insert({reorder[to], reorder[from]});
    }
    input_file.close();
	return {id, input};
}

void save_answer(const std::string &output_file_name, uint32_t *answer, int k) {
    std::fstream output_file;
    output_file.open(output_file_name, std::fstream::out);
    if(!output_file) {
        std::cout << "Printing output:" << std::endl;
        for(int i = 1; i <= k; ++i) {
            std::cout << answer[i] << " ";
        }
        std::cout << std::endl;
        error("Couldn't open: " + output_file_name);
    }
    for(int i = 1; i <= k; ++i) {
        output_file << answer[i] << " ";
    }
    output_file.close();
}
#endif //UTILS_H