#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

// Used for finding overlapping user ids from Reddit Corpus (100 highly active subreddits)
// Same list of subreddits as Reddit Corpus (small)
// https://convokit.cornell.edu/documentation/subreddit.html

// ./convokit_overlap_finder ~/.convokit/saved-corpora/ ~/GraphZeppelin/datasets/reddit/others/subreddits_small_sample.txt ~/GraphZeppelin/CUDA/GraphZeppelin/bipartite_datasets/

struct SubredditData {
  std::string subreddit_id;
  std::vector<std::string> user_ids;
  std::vector<int> overlap_user_ids;
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

std::vector<int> get_overlap_user_ids(std::vector<SubredditData> subreddits, SubredditData cur_subreddit) {
  std::vector<int> overlap_user_ids;

  std::cout << "Finding overlapping user ids for subreddit: " << cur_subreddit.subreddit_id << "\n";

  for (auto& subreddit : subreddits) {
    if (subreddit.subreddit_id == cur_subreddit.subreddit_id) {
      overlap_user_ids.push_back(-1);
      continue;
    }

    std::vector<std::string> global_user_ids;
    int total_num_users = cur_subreddit.user_ids.size() + subreddit.user_ids.size();
    global_user_ids.reserve(total_num_users);

    // Insert user ids from both subreddit
    global_user_ids.insert(global_user_ids.end(), cur_subreddit.user_ids.begin(), cur_subreddit.user_ids.end());
    global_user_ids.insert(global_user_ids.end(), subreddit.user_ids.begin(), subreddit.user_ids.end());

    // Remove any duplicate edges
    std::sort(global_user_ids.begin(), global_user_ids.end());
    global_user_ids.erase(std::unique(global_user_ids.begin(), global_user_ids.end()), global_user_ids.end());

    overlap_user_ids.push_back(total_num_users - global_user_ids.size());
  } 
  return overlap_user_ids;
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

  size_t total_num_users = 0;
  for (auto& subreddit: subreddits) {
    total_num_users += subreddit.user_ids.size();
  }
  std::cout << "Total Number Users: " << total_num_users << std::endl;

  timer_start = std::chrono::steady_clock::now();

  // Find overlaps
  #pragma omp parallel for
  for (auto& subreddit : subreddits) {
    subreddit.overlap_user_ids = get_overlap_user_ids(subreddits, subreddit);
  }

  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Finished getting overlaps: " << duration.count() << "s " <<  std::endl;

  // Write output file
  std::ofstream output_file(output_dir + "reddit_100active_corpus_duplicates.csv");

  // Write header
  output_file << "Subreddit,";
  std::string sep = "";
  for (auto& subreddit: subreddits) {
    output_file << sep << subreddit.subreddit_id;
    sep = ",";
  }
  output_file << "\n";

  // Write body
  for (auto& subreddit: subreddits) {
    sep = "";
    output_file << subreddit.subreddit_id << ",";
    for (auto& num_overlaps : subreddit.overlap_user_ids) {
      output_file << sep << num_overlaps;
      sep = ",";
    }
    output_file << "\n";
  }

  output_file.close();

  return 0;
}