#define BLOCK_SIZE 256
// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 8192

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <ctime>
#include <cstdio>

void sequential_bfs(const int *nodePtrs, const int *nodeNeighbors, int* nodeVisited, const int* currLevelNodes, int *nextLevelNodes, const int numCurrLevelNodes, int &numNextLevelNodes ){
	for(int i=0;i<numCurrLevelNodes;i++){
		auto node=currLevelNodes[i];
		for(auto nbrI=nodePtrs[node];nbrI<nodePtrs[node+1];nbrI++){
			auto neighbor=nodeNeighbors[nbrI];
			if(!nodeVisited[neighbor]){
				nodeVisited[neighbor]=1;
				nextLevelNodes[numNextLevelNodes++]=neighbor;
			}
		}
	}
}

__global__ void cuda_simple_bfs(int *nodePtrs, int *nodeNeighbors, int* nodeVisited, int* currLevelNodes, int *nextLevelNodes, const int numCurrLevelNodes, int *numNextLevelNodes ){

	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<numCurrLevelNodes){
		auto nbrI =nodePtrs[currLevelNodes[idx]  ];
		auto nbrI1=nodePtrs[currLevelNodes[idx]+1];
		while(nbrI++<nbrI1){
			auto neighbor=nodeNeighbors[nbrI];
			if(!atomicCAS(&(nodeVisited[neighbor]),0,1)){
				auto oldPos=atomicAdd(numNextLevelNodes,1);
				nextLevelNodes[oldPos]=neighbor;
			}
		}
	}

}

__global__ void cuda_optimized_bfs(int *nodePtrs, int *nodeNeighbors, int *nodeVisited, int* currLevelNodes, int *nextLevelNodes, const int numCurrLevelNodes, int *numNextLevelNodes){

	__shared__ int localNumNextLevelNodes[1];
	localNumNextLevelNodes[0]=0;
	__shared__ int localNextLevelNodes[BQ_CAPACITY];

	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<numCurrLevelNodes){
		auto nbrI =nodePtrs[currLevelNodes[idx]  ];
		auto nbrI1=nodePtrs[currLevelNodes[idx]+1];
		while(nbrI++<nbrI1){
			auto neighbor=nodeNeighbors[nbrI];
			if(!atomicCAS(&(nodeVisited[neighbor]),0,1)){
				auto oldPos=atomicAdd(localNumNextLevelNodes,1);
				if(oldPos<BQ_CAPACITY){
					localNextLevelNodes[oldPos]=neighbor;	
				}
				else{
					atomicAdd(localNumNextLevelNodes,-1);
					auto oldGlobalPos=atomicAdd(numNextLevelNodes,1);
					nextLevelNodes[oldGlobalPos]=neighbor;
				}
			}
		}
	}
	__syncthreads();

	int i=threadIdx.x;
	while(i<localNumNextLevelNodes[0]){

		auto oldPos=atomicAdd(numNextLevelNodes,1);
		nextLevelNodes[oldPos]=localNextLevelNodes[i];

		i+=blockDim.x;
	}

}


void read_file(const char* fileName, int& numNodes, int& numEdges, std::vector<std::vector<int>>& nodeNeighbors){

	std::fstream inputFile;

	inputFile.open(fileName,std::ios::in);

	if(inputFile)
		std::cout<<"File opened correctly"<<std::endl;

	else
		std::cout<<"File not opened"<<std::endl;

	inputFile>>numNodes;
	inputFile>>numEdges;

	int node=0;
	int neighbor=0;

	nodeNeighbors.reserve(numNodes);

	for(int i=0;i<numNodes;i++)
		nodeNeighbors.push_back(std::vector<int>());

	for(int i=0;i<numEdges;i++){
		inputFile>>node;
		inputFile>>neighbor;

		//Add the neighbor to the node
		nodeNeighbors[node].push_back(neighbor);
		//Add the node to the neighbor
		nodeNeighbors[neighbor].push_back(node);

	}

}

void initialize_sequential_collections(const int numNodes, const int numEdges, const std::vector<std::vector<int>>& _nodeNeighbors, int* nodeNeighbors, int* nodePtrs, int* visitedNodes, int* currLevelNodes, int& numCurrLevelNodes, int* nextLevelNodes, int& numNextLevelNodes){

	int iter=0;

	for(size_t i=0;i<_nodeNeighbors.size();i++){
		nodePtrs[i]=iter;
		for(size_t j=0;j<_nodeNeighbors[i].size();j++){
			nodeNeighbors[iter++]=_nodeNeighbors[i][j];
		}
	}

	nodePtrs[numNodes]=numEdges;

	memset(visitedNodes, 0, sizeof(int)*numNodes);

	visitedNodes[0]=1;

	memset(currLevelNodes, 0, sizeof(int)*numNodes);

	currLevelNodes[0]=0;

	numCurrLevelNodes=1;

	numNextLevelNodes=0;

}

void check_cuda_error(cudaError_t err){
	if(err){
		std::cerr<<"Cuda error: "<<err<<std::endl;
		std::cerr<<cudaGetErrorString(err)<<std::endl;
	}
}

int main(int argc, char** argv){

  int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		std::cout<<"Device Number: "<<i<<std::endl;
		std::cout<<"Device Number: "<<prop.name<<std::endl;
		std::cout<<"  max Blocks Per MultiProcessor: "<<prop.maxBlocksPerMultiProcessor<<std::endl;
		std::cout<<"  max Threads Per MultiProcessor: "<<prop.maxThreadsPerMultiProcessor<<std::endl;
		std::cout<<"  max Threads Per Block: "<<prop.maxThreadsPerBlock<<std::endl;
		std::cout<<"  num SM: "<<prop.multiProcessorCount<<std::endl;
		std::cout<<"  num bytes sharedMem Per Block: "<<prop.sharedMemPerBlock<<std::endl;
		std::cout<<"  num bytes sharedMem Per Multiprocessor: "<<prop.sharedMemPerMultiprocessor<<std::endl;
		std::cout<<"  Memory Clock Rate (KHz): "<<prop.memoryClockRate<<std::endl;
		std::cout<<"  Memory Bus Width (bits): "<<prop.memoryBusWidth<<std::endl;
		std::cout<<"  Peak Memory Bandwidth (GB/s): "<<2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6<<std::endl;
	}

	int numNodes=0;
	int numEdges=0;

	clock_t seq_start,seq_end;
	double seq_time=0,seq_partial_time=0;

	std::vector<std::vector<int>> _nodeNeighbors;

	if(argc!=1){
		read_file(argv[1], numNodes, numEdges, _nodeNeighbors);
	}
	else{
		read_file("drive/MyDrive/PARALLEL_COMPUTING_CHALLENGES/standard.txt", numNodes, numEdges, _nodeNeighbors);
	}

	//Construct the arrays
	int* nodeNeighbors=new int[numEdges<<1];
	int* nodePtrs=new int[numNodes+1];
	int* visitedNodes=new int[numNodes];
	int* currLevelNodes=new int[numNodes];
	int numCurrLevelNodes=1;
	int* nextLevelNodes=new int[numNodes];
	int numNextLevelNodes=0;

	initialize_sequential_collections(numNodes, numEdges, _nodeNeighbors, nodeNeighbors, nodePtrs, visitedNodes, currLevelNodes, numCurrLevelNodes, nextLevelNodes, numNextLevelNodes);

	int iter=0;

	std::cout<<"--------------------------------------------------------"<<std::endl;
	std::cout<<"SEQUENTIAL ALGORITHM"<<std::endl;
	std::cout<<"--------------------------------------------------------"<<std::endl;

	do{

		memset(nextLevelNodes, 0, sizeof(int)*numNodes);
		numNextLevelNodes=0;

		seq_start=clock();

		sequential_bfs(nodePtrs, nodeNeighbors, visitedNodes, currLevelNodes, nextLevelNodes, numCurrLevelNodes, numNextLevelNodes);

		seq_end=clock();

		seq_partial_time=((double)(seq_end-seq_start))/CLOCKS_PER_SEC;

		seq_time+=seq_partial_time;

		std::cout<<"Level: "<<iter++<<std::endl;
		std::cout<<"Current level nodes: "<<numCurrLevelNodes<<std::endl;
		std::cout<<"Next    level nodes: "<<numNextLevelNodes<<std::endl;
		/*
			 for(int i=0;i<numNextLevelNodes;i++)
			 std::cout<<nextLevelNodes[i]<<" ";
			 std::cout<<std::endl;
		 */

		std::cout<<"Time taken (s): "<<seq_partial_time<<std::endl;

		std::swap(nextLevelNodes,currLevelNodes);

		numCurrLevelNodes=numNextLevelNodes;

	}while(numNextLevelNodes);

	std::cout<<"--------------------------------------------------------"<<std::endl;
	std::cout<<"PARALLEL ALGORITHM"<<std::endl;
	std::cout<<"--------------------------------------------------------"<<std::endl;

	std::cout<<std::endl;

	cudaError_t err;
	cudaEvent_t cuda_start,cuda_stop;
	err=cudaEventCreate(&cuda_start); check_cuda_error(err);
	err=cudaEventCreate(&cuda_stop ); check_cuda_error(err);
	float cuda_elapsed_time_ms=0;
	double cuda_total_elapsed_time_ms=0;
	int num_blocks=0;

	err=cudaEventRecord(cuda_start, 0); check_cuda_error(err);

	int* cuda_nodeNeighbors, *cuda_nodePtrs, *cuda_visitedNodes, *cuda_currLevelNodes, *cuda_nextLevelNodes;
	int* cuda_numNextLevelNodes;

	err=cudaMallocManaged((void**)&cuda_nodeNeighbors , sizeof(int)*numEdges*2  ); check_cuda_error(err);
	err=cudaMallocManaged((void**)&cuda_nodePtrs      , sizeof(int)*(numNodes+1)); check_cuda_error(err);
	err=cudaMallocManaged((void**)&cuda_visitedNodes  , sizeof(int)*numNodes    ); check_cuda_error(err);
	err=cudaMallocManaged((void**)&cuda_currLevelNodes, sizeof(int)*numNodes    ); check_cuda_error(err);
	err=cudaMallocManaged((void**)&cuda_nextLevelNodes, sizeof(int)*numNodes    ); check_cuda_error(err);

	err=cudaMallocManaged((void**)&cuda_numNextLevelNodes, sizeof(int)); check_cuda_error(err);

	err=cudaMemcpy(cuda_nodeNeighbors , nodeNeighbors, sizeof(int)*numEdges*2  , cudaMemcpyHostToDevice); check_cuda_error(err);
	err=cudaMemcpy(cuda_nodePtrs      ,      nodePtrs, sizeof(int)*(numNodes+1), cudaMemcpyHostToDevice); check_cuda_error(err);
	err=cudaMemset(cuda_visitedNodes  ,             0, sizeof(int)*numNodes                            ); check_cuda_error(err);
	err=cudaMemset(cuda_currLevelNodes,             0, sizeof(int)*numNodes                            ); check_cuda_error(err);
	err=cudaMemset(cuda_nextLevelNodes,             0, sizeof(int)*numNodes                            ); check_cuda_error(err);

	cuda_visitedNodes[0]=1;
	cuda_currLevelNodes[0]=0;
	numCurrLevelNodes=1;

	err=cudaEventRecord(cuda_stop, 0); check_cuda_error(err);
	err=cudaEventSynchronize(cuda_stop); check_cuda_error(err);

	err=cudaEventElapsedTime(&cuda_elapsed_time_ms, cuda_start, cuda_stop); check_cuda_error(err);

	std::cout<<"Time taken to create data on gpu (ms): "<<cuda_elapsed_time_ms<<std::endl;

	iter=0;

	do{

		err=cudaMemset(cuda_nextLevelNodes,    0, sizeof(int)*numNodes); check_cuda_error(err);

		err=cudaMemset(cuda_numNextLevelNodes, 0, sizeof(int)         ); check_cuda_error(err);

		num_blocks=numCurrLevelNodes/BLOCK_SIZE+1;

		cudaEventRecord(cuda_start, 0);

		cuda_simple_bfs<<<num_blocks,BLOCK_SIZE>>>(cuda_nodePtrs, cuda_nodeNeighbors, cuda_visitedNodes, cuda_currLevelNodes, cuda_nextLevelNodes, numCurrLevelNodes, cuda_numNextLevelNodes);

		cudaEventRecord(cuda_stop, 0);

		err=cudaEventSynchronize(cuda_stop); check_cuda_error(err);

		err=cudaEventElapsedTime(&cuda_elapsed_time_ms, cuda_start, cuda_stop); check_cuda_error(err);

		cuda_total_elapsed_time_ms+=cuda_elapsed_time_ms;

		std::cout<<"Level: "<<iter++<<std::endl;
		std::cout<<"Number  of blocks: "<<num_blocks<<std::endl;
		std::cout<<"Threads per block: "<<BLOCK_SIZE<<std::endl;
		std::cout<<"Current level nodes: "<<numCurrLevelNodes<<std::endl;
		std::cout<<"Next    level nodes: "<<*cuda_numNextLevelNodes<<std::endl;

		/*
			 for(int i=0;i<*cuda_numNextLevelNodes;i++)
			 std::cout<<cuda_nextLevelNodes[i]<<" ";
			 std::cout<<std::endl;
		 */

		std::cout<<"Time taken (ms): "<<cuda_elapsed_time_ms<<std::endl;

		std::swap(cuda_nextLevelNodes,cuda_currLevelNodes);

		numCurrLevelNodes=*cuda_numNextLevelNodes;

	}while(numCurrLevelNodes);

	std::cout<<"--------------------------------------------------------"<<std::endl;
	std::cout<<"Error checking with the two methods: ";

	int check=0;

	for(int i=0;i<numNodes;i++)
		check+=visitedNodes[i]-cuda_visitedNodes[i];

	std::cout<<check<<std::endl;

	std::cout<<"--------------------------------------------------------"<<std::endl;
	std::cout<<"PARALLEL ALGORITHM OPTIMIZED"<<std::endl;
	std::cout<<"--------------------------------------------------------"<<std::endl;

	double cuda_optimized_total_elapsed_time_ms=0;

	err=cudaMemset(cuda_visitedNodes  ,             0, sizeof(int)*numNodes                            ); check_cuda_error(err);
	err=cudaMemset(cuda_currLevelNodes,             0, sizeof(int)*numNodes                            ); check_cuda_error(err);
	err=cudaMemset(cuda_nextLevelNodes,             0, sizeof(int)*numNodes                            ); check_cuda_error(err);

	cuda_visitedNodes[0]=1;
	cuda_currLevelNodes[0]=0;
	numCurrLevelNodes=1;

	iter=0;

	do{

		err=cudaMemset(cuda_nextLevelNodes,    0, sizeof(int)*numNodes); check_cuda_error(err);

		err=cudaMemset(cuda_numNextLevelNodes, 0, sizeof(int)         ); check_cuda_error(err);

		num_blocks=numCurrLevelNodes/BLOCK_SIZE+1;

		cudaEventRecord(cuda_start, 0);

		cuda_optimized_bfs<<<num_blocks,BLOCK_SIZE>>>(cuda_nodePtrs, cuda_nodeNeighbors, cuda_visitedNodes, cuda_currLevelNodes, cuda_nextLevelNodes, numCurrLevelNodes, cuda_numNextLevelNodes);

		cudaEventRecord(cuda_stop, 0);

		err=cudaEventSynchronize(cuda_stop); check_cuda_error(err);

		err=cudaEventElapsedTime(&cuda_elapsed_time_ms, cuda_start, cuda_stop); check_cuda_error(err);

		cuda_optimized_total_elapsed_time_ms+=cuda_elapsed_time_ms;

		std::cout<<"Level: "<<iter++<<std::endl;
		std::cout<<"Number  of blocks: "<<num_blocks<<std::endl;
		std::cout<<"Threads per block: "<<BLOCK_SIZE<<std::endl;
		std::cout<<"Current level nodes: "<<numCurrLevelNodes<<std::endl;
		std::cout<<"Next    level nodes: "<<*cuda_numNextLevelNodes<<std::endl;
		/*
		for(int i=0;i<*cuda_numNextLevelNodes;i++)
			std::cout<<cuda_nextLevelNodes[i]<<" ";
		 std::cout<<std::endl;
		*/

		std::cout<<"Time taken (ms): "<<cuda_elapsed_time_ms<<std::endl;

		std::swap(cuda_nextLevelNodes,cuda_currLevelNodes);

		numCurrLevelNodes=*cuda_numNextLevelNodes;

	}while(numCurrLevelNodes);
	std::cout<<"--------------------------------------------------------"<<std::endl;
	std::cout<<"Error checking with the two methods: ";
	check=0;

	for(int i=0;i<numNodes;i++)
		check+=visitedNodes[i]-cuda_visitedNodes[i];

	std::cout<<check<<std::endl;
	std::cout<<"--------------------------------------------------------"<<std::endl;
	std::cout<<"Total time taken by sequential         (s): "<<seq_time<<std::endl;
	std::cout<<"Total time taken by parallel standard  (s): "<<cuda_total_elapsed_time_ms/1000<<std::endl;
	std::cout<<"Total time taken by parallel optimized (s): "<<cuda_optimized_total_elapsed_time_ms/1000<<std::endl;
	std::cout<<"--------------------------------------------------------"<<std::endl;
	std::cout<<"Speedup sequential        / parallel standard : "<<seq_time/(cuda_total_elapsed_time_ms/1000)<<std::endl;
	std::cout<<"Speedup sequential        / parallel optimized: "<<seq_time/(cuda_optimized_total_elapsed_time_ms/1000)<<std::endl;
	std::cout<<"Speedup parallel standard / parallel optimized: "<<cuda_total_elapsed_time_ms/cuda_optimized_total_elapsed_time_ms<<std::endl;
  std::cout<<"--------------------------------------------------------"<<std::endl;


	delete[] nodeNeighbors;
	delete[] nodePtrs;
	delete[] visitedNodes;
	delete[] currLevelNodes;
	delete[] nextLevelNodes;

	err=cudaFree(cuda_nodeNeighbors ); check_cuda_error(err);
	err=cudaFree(cuda_nodePtrs      ); check_cuda_error(err);
	err=cudaFree(cuda_visitedNodes  ); check_cuda_error(err);
	err=cudaFree(cuda_currLevelNodes); check_cuda_error(err);
	err=cudaFree(cuda_nextLevelNodes); check_cuda_error(err);

	return 0;
}
