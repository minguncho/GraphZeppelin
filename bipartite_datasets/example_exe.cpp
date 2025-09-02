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

void project_dataset(string filename) {

  Interactions interactions;
  interactions = parse_file(filename);

  Eigen::SparseMatrix<double, Eigen::ColMajor> bipartite_sparse_matrix(interactions.num_users, interactions.num_subreddits);
  int counter = 0;
  for (Interaction i : interactions.interaction_vec) {
    bipartite_sparse_matrix.coeffRef(i.userID, i.subredditID) = 1;
  }

  //std::cout << bipartite_sparse_matrix << std::endl;
  Eigen::SparseMatrix<double, Eigen::ColMajor> transposed_bipartite_sparse_matrix =  bipartite_sparse_matrix.transpose();
  //std::cout << transposed_bipartite_sparse_matrix << std::endl;
  std::cout << "This message should print" << std::endl;
  Eigen::SparseMatrix<double, Eigen::ColMajor> projected = bipartite_sparse_matrix * transposed_bipartite_sparse_matrix;
  std::cout << "This message shouldn't print" << std::endl;
  //std::cout << projected << std::endl;
  printf("Projected matrix is %ld by %ld with %ld nonzeros \n", projected.outerSize(), projected.innerSize(), projected.nonZeros());

}

 
int main() {

  //string filename = "data/example.txt";
  string filename = "data/kaggle_reddit_2021_cleaned.txt";
  project_dataset(filename);



  // Eigen::SparseMatrix<double, Eigen::ColMajor> sm(10,5);
  // sm.coeffRef(0, 0) = 3;
  // sm.coeffRef(1, 0) = 2.5;
  // sm.coeffRef(0, 1) = -1;
  // sm.coeffRef(1, 1) = sm.coeffRef(1, 0) + sm.coeffRef(0, 1);
  // for (int i = 1; i < 5; i++) {
  //   sm.coeffRef(i*2, i) = 1.5;
  // }
  // //std::cout << sm << std::endl;
  // Eigen::SparseMatrix<double, Eigen::ColMajor> tsm =  sm.transpose();
  // //std::cout << tsm << std::endl;
  // Eigen::SparseMatrix<double, Eigen::ColMajor> projected = sm * tsm;
  // //std::cout << projected << std::endl;
  // printf("Projected matrix is %ld by %ld with %ld nonzeros \n", projected.outerSize(), projected.innerSize(), projected.nonZeros());
}
