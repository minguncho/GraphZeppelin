#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <nlohmann/json.hpp>

// Used for parsing information from Reddit Corpus (100 highly active subreddits)
// Same list of subreddits as Reddit Corpus (small)
// https://convokit.cornell.edu/documentation/subreddit.html

// ./convokit_parser ~/.convokit/saved-corpora/ ~/GraphZeppelin/datasets/reddit/others/subreddits_small_sample.txt ~/GraphZeppelin/CUDA/GraphZeppelin/bipartite_datasets/
// ./convokit_parser ~/.convokit/saved-corpora/ ~/GraphZeppelin/datasets/reddit/others/subreddits_small_sample_200k.txt ~/GraphZeppelin/CUDA/GraphZeppelin/bipartite_datasets/

struct SubredditData {
  std::unordered_map<std::string, int> user_ids;
  std::vector<std::string> conv_data;
};

// User IDs to exclude while parsing
std::unordered_set<std::string> filtered_userids = {
  "[deleted]", "AutoModerator"
};

std::unordered_map<std::string, int> read_subreddit_file(std::string subreddit_list_file) {
  std::unordered_map<std::string, int> subreddit_ids;
  std::ifstream file(subreddit_list_file);

  if (file.is_open()) {
    std::string line;
    int cur_subreddit_id = 0; 
    while (getline(file, line)) {
      subreddit_ids[line] = cur_subreddit_id;
      cur_subreddit_id++;
    }
  }
  else {
    std::cerr << "Error: Unable to open subreddit list file: " << subreddit_list_file << std::endl;
    exit(EXIT_FAILURE);
  }

  return subreddit_ids;
}

SubredditData parse_data(std::string dataset_dir, std::unordered_map<std::string, int> subreddit_ids) {
  std::unordered_map<std::string, int> user_ids;
  std::vector<std::string> conv_data;

  // Iterate through each subreddit
  int cur_user_id = 0; 
  for (auto& subreddit : subreddit_ids) {
    // Open users.json file 
    std::string file_path = dataset_dir + subreddit.first + "/users.json";
    std::ifstream file(file_path);

    std::cout << "Processing users for subreddit: " << subreddit.first << "\n";

    if (file.is_open()) {
      // Parse corpus file into json 
      nlohmann::json users_json = nlohmann::json::parse(file);
      
      for (auto& [user_id, info] : users_json.items()) {
        if (filtered_userids.find(user_id) != filtered_userids.end()) continue;

        // Collect user ids
        if (user_ids.find(user_id) == user_ids.end()) {
          user_ids[user_id] = cur_user_id;
          cur_user_id++;
        }

        // Save data
        conv_data.push_back(std::to_string(user_ids[user_id]) + ", " + std::to_string(subreddit.second));
      }
      
    }
    else {
      std::cerr << "Error: Unable to open users file: " << file_path << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  std::cout << "Total number of unique users: " << user_ids.size() << "\n";
  return {user_ids, conv_data};
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

  auto timer_start = std::chrono::steady_clock::now();
  std::unordered_map<std::string, int> subreddit_ids = read_subreddit_file(subreddit_list_file);
  std::chrono::duration<double> duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Finished reading subreddit file: " << duration.count() << "s " <<  std::endl;

  timer_start = std::chrono::steady_clock::now();
  SubredditData subredditData = parse_data(dataset_dir, subreddit_ids);
  duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Finished parsing data: " << duration.count() << "s " <<  std::endl;

  // Write output file
  std::ofstream output_file(output_dir + "reddit_100active_RENAME_corpus.txt");

  // Write header
  output_file << std::to_string(subredditData.user_ids.size()) + ", " + 
                 std::to_string(subreddit_ids.size()) + ", " + 
                 std::to_string(subredditData.conv_data.size()) + "\n";

  for (auto& conv : subredditData.conv_data) {
    output_file << conv << "\n";
  }

  output_file.close();

  return 0;
}