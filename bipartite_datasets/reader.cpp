#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
using namespace std;

struct Interaction {
    int userID;
    int subredditID;
};

std::vector<Interaction> parse_file(string filename) {
    string line;
    ifstream myfile (filename);
    std::vector<Interaction> interactions;
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

        printf("Dataset has %d users, %d subreddits, and %d interactions \n", num_users, num_subreddits, num_lines);

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

            interactions.push_back(interaction);
        }
        myfile.close();

        // for (const auto& i : interactions) {
        //     std::cout << i.userID << ", " << i.subredditID << std::endl;
        // }
        printf("file has %lu interactions\n", interactions.size());
    }

    else cout << "Unable to open file\n"; 
    return interactions;
}

int main(){
    string filename = "data/kaggle_reddit_2021_cleaned.txt";
    parse_file(filename);
}