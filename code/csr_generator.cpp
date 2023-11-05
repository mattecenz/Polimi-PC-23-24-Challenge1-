#include <iostream> 
#include <unordered_set>
#include <random>
#include <cstdint>
#include <algorithm>
#include <ctime>

/*
To compile:

g++ -Wall csr_generator.cpp -o generator 

Can use also -02 (when creating very big graphs, like more than 10^6 nodes, can make a difference

To execute:

./generator [N_VERTEX] [GENERATION_CONSTANT] [SEED] > out.txt

(they are optional but they have to be in that exact order)

Where:

N_VERTEX            -> number of vertices of the graph

GENERATION_CONSTANT -> number of edges created (approximately) for every node

SEED                -> seed for the random number generator 

*/
int main(int argc, char** argv){

	uint32_t n_vertex=1000;
	uint32_t generation_constant=10;
	uint32_t seed=12345678;

	//Function to read arguments
	if(argc!=1){

		switch(argc){
			case 4:

				seed=std::atoi(argv[3]);

			case 3:

				generation_constant=std::atoi(argv[2]);

			case 2:

				n_vertex=std::atoi(argv[1]);

				if(n_vertex<=generation_constant){
					std::cerr<<"Conditions ill-posed"<<std::endl;
					return 0;
				}

				break;
			default:
				std::cerr<<"Invalid number of arguments >:("<<std::endl;
				return 0;
				break;
		}

	}

	clock_t start=clock();

	//Create a set of edges (pairs of numbers)
	std::unordered_set<uint64_t> edges_set;	

	std::default_random_engine rd{seed};

	for(uint32_t i=0;i<n_vertex;i++){

		for(uint32_t j=0;j<generation_constant;j++){

			//Generate another random node
			uint32_t num=rd()%n_vertex;

			//No loops in graph
			while(num==i) num=rd()%n_vertex;

			if(i<num) edges_set.insert((uint64_t)(((uint64_t)i)<<32)+(uint64_t)num);

			else edges_set.insert((uint64_t)(((uint64_t)num)<<32)+(uint64_t)i);

		}
	}

	//Now you have to sort
	std::vector<uint64_t> ordered_edges(edges_set.size());

	ordered_edges.reserve(edges_set.size());

	std::copy(edges_set.begin(),edges_set.end(), ordered_edges.begin());

	std::sort(ordered_edges.begin(), ordered_edges.end(), 
			[](const uint64_t &a, const uint64_t &b){
			uint32_t na=a>>32;
			uint32_t nb=b>>32;
			if(na!=nb) return na<nb;
			na=a-((a>>32)<<32);
			nb=b-((b>>32)<<32);
			return na<nb;
			}
			);

	//Remove one edge to make it not fully connected
	//uint32_t rand=rd%(n_vertices-generation_constant*2)+generation_constant*2;
	//ordered_edges.erase(rand);

	std::cout<<n_vertex<<std::endl;
	std::cout<<ordered_edges.size()<<std::endl;

	for(auto i: ordered_edges){
		std::cout<<(i>>32)<<" "<<i-((i>>32)<<32)<<std::endl;
	}

	clock_t end=clock();

	//std::cout<<"Time: "<<((double)(end-start))/CLOCKS_PER_SEC<<std::endl;

	return 0;
}	
