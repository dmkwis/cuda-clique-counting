CC = /usr/local/cuda/bin/nvcc
CFLAGS = -O3 -std=c++17 -Xcompiler -Wall -Xcompiler -Wextra -arch=sm_70 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70
binaries = kcliques
headers = utils.h types.h orientation_count.h preprocessing.h direct_edges.h vertex_ordering.h

all: kcliques

kcliques: solution.cu $(headers)
	$(CC) $(CFLAGS) $< -o $@

.PHONY: clean

clean:
	rm $(binaries)