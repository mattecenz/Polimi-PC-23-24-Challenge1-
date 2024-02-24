# Polimi-PC-23-24-Challenge1-

The challenge presented in the course **Parallel Computing** at *Politecnico di Milano* consisted in an implementation of the classical BFS search in a sparse graph using both a CPU and a GPU implementation (with CUDA).

## Introduction and main objectives

The main goal for this problem is to view the performance increase in a massively parallel environment when utilizing dedicated accelerators.

The algorithm implemented is an incremental search in the graph starting from the node 0. So at the first step all the neighbours of 0 are visited, then the neighbours of the neighbours and so on...

The algorithm is implemented in a way that at each step both the CPU and GPU kernel produce the same output, the new listed of visited nodes and the new neighbours to check.

## Test graphs 

The code supports the reading of a file in CSR format, with the first two rows of the files being the total number of nodes in the graph and the total number of edges. **REMEMBER** We assume here an undirected graph.

In order to have more flexibility a dedicated script has been created (called *csr_generator.cpp* ) which can be compiled with your preferred compiler and executed with:

`./generator_name [N_VERTEX] [GENERATION_CONSTANT] [SEED] > out.txt`

Where the generation constant represents how many edges are created for each node on average.

**NOTE:** the file generated is in ascending order with respect to the node number.

# Running the program

In order to run the program we can compile the file using NVCC :

`nvcc exercise_v1.cu - o exercise`

And simply run it with:

`./exercise [PATH_TO_FILE]`

The program shows at the end what has been the total time taken to run the kernels and shows the different results obtained.
If for some reason there should be an error when running the CUDA kernels it will be prompted and from that moment all the rest of the outputs should be invalidated.
