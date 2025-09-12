#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Sparse>
using namespace std;

struct Interaction {
    int userID;
    int subredditID;
};

struct Interactions {
  std::vector<Interaction> interaction_vec;
  int num_users;
  int num_subreddits;
};

//sorting interactions in this order minimizes cache misses for insertion in column-major sparse matrix format.
bool compareInteraction(const Interaction &a, const Interaction &b) {
  if (a.subredditID == b.subredditID) {
    return a.userID < b.userID;
  }
  else {
    return a.subredditID < b.subredditID;
  }
}

Interactions parse_file(string filename) {
    string line;
    ifstream myfile (filename);
    std::vector<Interaction> interactions;
    Interactions output;
    if (myfile.is_open())
    {
        //handle the header line
        getline(myfile, line);

        istringstream ss(line);
        string num_users_str;
        getline(ss, num_users_str, ',');
        string num_subreddits_str;
        getline(ss, num_subreddits_str, ',');
        string num_lines_str;
        getline(ss, num_lines_str, '\n');
        int num_users = stoi(num_users_str);
        int num_subreddits = stoi(num_subreddits_str);
        int num_lines = stoi(num_lines_str);

        output.interaction_vec = interactions;
        output.num_users = num_users;
        output.num_subreddits = num_subreddits;

        printf("Dataset has %d users, %d subreddits, and %d interactions \n", output.num_users, output.num_subreddits, num_lines);

        //loop over remaining lines
        while ( getline (myfile,line) )
        {
            istringstream ss(line);
            Interaction interaction;
            string userIDstr;
            string subredditIDstr;
            getline(ss, userIDstr, ',');
            interaction.userID = stoi(userIDstr);
            getline(ss, subredditIDstr, '\n');
            interaction.subredditID = stoi(subredditIDstr);

            output.interaction_vec.push_back(interaction);
        }
        myfile.close();

        // for (const auto& i : interactions) {
        //     std::cout << i.userID << ", " << i.subredditID << std::endl;
        // }
        printf("file has %lu interactions\n", output.interaction_vec.size());
    }

    else cout << "Unable to open file\n"; 

    std::sort(output.interaction_vec.begin(), output.interaction_vec.end(), compareInteraction);
    std::cout << "parsed and sorted interactions." << std::endl;
    return output;
}

void project_dataset(string filename, bool display = false) {

  Interactions interactions;
  interactions = parse_file(filename);

  int rows = interactions.num_users;
  int cols = interactions.num_subreddits;

  Eigen::SparseMatrix<int, Eigen::ColMajor, long long int> bipartite_sparse_matrix(rows, cols);
  //int counter = 0;
  for (Interaction i : interactions.interaction_vec) {
    bipartite_sparse_matrix.coeffRef(i.userID, i.subredditID) = 1;
  }

  //auto result = bipartite_sparse_matrix.colwise().sum();

  Eigen::VectorXi ones = Eigen::VectorXi::Ones(rows);
  Eigen::VectorXi result = bipartite_sparse_matrix.transpose() * ones;
  double sum = result.sum();
  double mean = sum / (static_cast<double>(cols));
  printf("mean subreddit degree is %f over %d columns\n", mean, cols);
  
  long int edge_upper_bound = 0;
  int counter = 0;
  for (double i : result) {
    if (display && counter%1000==0) {
      printf("check 1 %ld\n", edge_upper_bound);
    }
    int degree = static_cast<int>(i);
    int clique_size = (degree * (degree-1))/2;
    edge_upper_bound += clique_size;
    if (display && counter%1000==0) {
      printf("clique size %d, check 2 %ld\n", clique_size, edge_upper_bound);
    }
    counter++;
  }
  printf("upper bound on number of edges: %ld\n", edge_upper_bound);

  printf("setting index type to be long long int which has %d bytes on this machine\n", sizeof(long long int));
  //std::cout << bipartite_sparse_matrix << std::endl;
  Eigen::SparseMatrix<int, Eigen::ColMajor, long long int> transposed_bipartite_sparse_matrix =  bipartite_sparse_matrix.transpose();
  //std::cout << transposed_bipartite_sparse_matrix << std::endl;
  std::cout << "Bipartite matrix and transpose produced. Computing projected matrix:" << std::endl;
  Eigen::SparseMatrix<int, Eigen::ColMajor, long long int> projected = bipartite_sparse_matrix * transposed_bipartite_sparse_matrix;
  //std::cout << projected << std::endl;
  printf("Projected matrix is %ld by %ld with %ld nonzeros \n", projected.outerSize(), projected.innerSize(), projected.nonZeros());

}

 
int main() {
  //string filename = "data/example.txt";
  string filename = "../data/kaggle_reddit_2021_cleaned.txt";
  project_dataset(filename);
}
