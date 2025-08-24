import json
import os
import sys

f_name = sys.argv[1]
with open(f_name) as f:
  # Dictionary for holding user_id and subreddit
  user_ids = {} # key: user_id (str), val: numerical_id (int)
  subreddit_ids = {} # key: subreddit_id (str), val: numerical_id (int)

  # Counter variables
  cur_user_id = 0
  cur_subreddit_id = 0
  num_lines = 0

  # Input lines converted into numerical id
  input_lines = []

  for line in f:
    data = json.loads(line)

    user_id = data['user_id']
    subreddit_id = data['subreddit']

    if user_id not in user_ids:
      user_ids[user_id] = cur_user_id
      cur_user_id += 1
    
    if subreddit_id not in subreddit_ids:
      subreddit_ids[subreddit_id] = cur_subreddit_id
      cur_subreddit_id += 1

    input_lines.append(str(user_ids[user_id]) + ', ' + str(subreddit_ids[subreddit_id]))

    num_lines += 1

  with open(os.path.splitext(f_name)[0] + '_cleaned.txt', 'w') as output_f:
    # Write the header with number of user_ids, subreddit_ids, and number of lines
    output_f.write(str(len(user_ids)) + ', ' + str(len(subreddit_ids)) + ', ' + str(num_lines) + '\n')
    for input_line in input_lines:
      output_f.write(input_line + '\n')

    
  