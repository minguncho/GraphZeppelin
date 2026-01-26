#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <nlohmann/json.hpp>

// Used for parsing conversation data from Reddit Corpus (small) Dataset
// https://convokit.cornell.edu/documentation/reddit-small.html

// User IDs to exclude while parsing
std::unordered_set<std::string> filtered_userids = {
  //"[deleted]", "AutoModerator"
};

struct SubredditData {
  std::unordered_map<std::string, int> user_ids;
  std::unordered_map<std::string, int> subreddit_ids;
  std::vector<std::string> utterances;
};

SubredditData parse_data(std::string utterance_file) {
  std::ifstream utt_file(utterance_file);

  // Check if input files exist and can be opened
  if (!utt_file.is_open()) {
    std::cerr << "Error: Unable to open utterance file: " << utterance_file << std::endl;
    exit(EXIT_FAILURE);
  }

  // Parse input files into json 
  nlohmann::json utt_json = nlohmann::json::parse(utt_file);

  // Maps for holding unique user and subreddit ids
  std::unordered_map<std::string, int> user_ids;
  std::unordered_map<std::string, int> subreddit_ids;
  std::vector<std::string> utterances;

  int cur_user_id = 0;
  int cur_subreddit_id = 0;

  // Iterate through utterance and conversation 
  for (auto utt_data = utt_json.begin(); utt_data != utt_json.end(); utt_data++) {
    std::string user_id = (*utt_data)["user"];
    std::string subreddit_id = (*utt_data)["meta"]["subreddit"];

    if (filtered_userids.find(user_id) != filtered_userids.end()) continue;

    // Get unique numerical id for user and subreddit
    if (user_ids.find(user_id) == user_ids.end()) {
      user_ids[user_id] = cur_user_id;
      cur_user_id++;
    }

    if (subreddit_ids.find(subreddit_id) == subreddit_ids.end()) {
      subreddit_ids[subreddit_id] = cur_subreddit_id;
      cur_subreddit_id++;
    }

    utterances.push_back(std::to_string(user_ids[user_id]) + ", " + std::to_string(subreddit_ids[subreddit_id]));
  }

  // Close input files
  utt_file.close();

  SubredditData subredditData;
  subredditData.user_ids = user_ids;
  subredditData.subreddit_ids = subreddit_ids;
  subredditData.utterances = utterances;

  return subredditData;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: utterance_file output_dir" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string utterance_file = argv[1];
  std::string output_dir = argv[2];

  auto timer_start = std::chrono::steady_clock::now();
  SubredditData subredditData = parse_data(utterance_file);

  std::chrono::duration<double> duration = std::chrono::steady_clock::now() - timer_start;
  std::cout << "Finished parsing utterance file: " << duration.count() << "s " <<  std::endl;

  std::cout << "Number of unique users: " << subredditData.user_ids.size() << "\n";
  std::cout << "Number of unique subreddits: " << subredditData.subreddit_ids.size() << "\n";
  std::cout << "Number of utterances (post, comment): " << subredditData.utterances.size() << "\n";

  // Write to output file
  std::ofstream output_file(output_dir + "reddit_small_corpus.txt");

  // Write header
  output_file << std::to_string(subredditData.user_ids.size()) + ", " + 
                 std::to_string(subredditData.subreddit_ids.size()) + ", " + 
                 std::to_string(subredditData.utterances.size()) + "\n";

  for (auto& utt : subredditData.utterances) {
    output_file << utt << "\n";
  }

  output_file.close();

  return 0;
}
