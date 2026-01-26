#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

// Used for getting upper bound limit (number of edges) for Reddit Corpus (100 highly active subreddits)
// Same list of subreddits as Reddit Corpus (small)
// https://convokit.cornell.edu/documentation/subreddit.html

// ./convokit_num_users ~/.convokit/saved-corpora/ ~/GraphZeppelin/datasets/reddit/others/subreddits_small_sample.txt ~/GraphZeppelin/CUDA/GraphZeppelin/bipartite_datasets/

struct SubredditData {
  std::string subreddit_id;
  std::vector<std::string> user_ids;
};

// User IDs to exclude while parsing
std::unordered_set<std::string> filtered_userids = {
  "[deleted]", "AutoModerator"
};

std::vector<std::string> read_subreddit_file(std::string subreddit_list_file) {
  std::vector<std::string> subreddit_ids;
  std::ifstream file(subreddit_list_file);

  if (file.is_open()) {
    std::string line;
    while (getline(file, line)) {
      subreddit_ids.push_back(line);
    }
  }
  else {
    std::cerr << "Error: Unable to open subreddit list file: " << subreddit_list_file << std::endl;
    exit(EXIT_FAILURE);
  }

  return subreddit_ids;
}

std::vector<std::string> get_user_ids(std::string dataset_dir, std::string subreddit_id) {
  std::vector<std::string> user_ids;

  // Open users.json file 
  std::string file_path = dataset_dir + subreddit_id + "/users.json";
  std::ifstream file(file_path);

  std::cout << "Processing users for subreddit: " << subreddit_id << "\n";

  if (file.is_open()) {
    // Parse corpus file into json 
    nlohmann::json users_json = nlohmann::json::parse(file);
    
    for (auto& [user_id, info] : users_json.items()) {
      if (filtered_userids.find(user_id) != filtered_userids.end()) continue;
      user_ids.push_back(user_id);
    }
    
  }
  else {
    std::cerr << "Error: Unable to open users file: " << file_path << std::endl;
    exit(EXIT_FAILURE);
  }
  
  return user_ids;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: dataset_dir subreddit_list_file output_dir" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string dataset_dir = argv[1];
  std::string subreddit_list_file = argv[2];
  std::string output_dir = argv[3];

  std::vector<SubredditData> subreddits;

  auto timer_start = std::chrono::steady_clock::now();
  std::vector<std::string> subreddit_ids = read_subreddit_file(subreddit_list_file);
  std::chrono::duration<double> duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Finished reading subreddit file: " << duration.count() << "s " <<  std::endl;

  timer_start = std::chrono::steady_clock::now();

  #pragma omp parallel for
  for (auto& subreddit_id : subreddit_ids) {
    SubredditData data;
    data.subreddit_id = subreddit_id;
    data.user_ids = get_user_ids(dataset_dir, subreddit_id);
    #pragma omp critical
    {
      subreddits.push_back(data);
    }
  }

  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Finished getting user ids: " << duration.count() << "s " <<  std::endl;
  std::cout << "Number of subreddts: " << subreddits.size() << std::endl;

  // Sort based on number of users
  std::sort(subreddits.begin(), subreddits.end(), 
    [](const SubredditData& a, const SubredditData& b) {
      return a.user_ids.size() > b.user_ids.size();
    });

  // Write output file
  std::ofstream output_file(output_dir + "reddit_100active_corpus_num_users.csv");

  // Write header
  output_file << "Subreddit,num_users\n";

  // Write body
  size_t check_edge = 1000000;
  size_t total_upper_edges = 0;
  int num_subreddits = 0;
  for (auto& subreddit: subreddits) {
    output_file << subreddit.subreddit_id << "," << subreddit.user_ids.size() << "\n";
    if (subreddit.user_ids.size() < check_edge) {
      size_t num_users = subreddit.user_ids.size();
      total_upper_edges += (subreddit.user_ids.size() * (subreddit.user_ids.size() - 1)) / 2;
      num_subreddits++;
    }
  }

  std::cout << "Checking for upper bound for subreddits with users < " << check_edge << std::endl;
  std::cout << "Number of subreddits: " << num_subreddits << std::endl;
  std::cout << "Total Upper Bound Edges: " << total_upper_edges << std::endl;

  output_file.close();

  

  return 0;
}