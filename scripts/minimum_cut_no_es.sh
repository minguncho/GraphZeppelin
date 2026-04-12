#
# Name: minimum_cut_no_es.sh
#
# Full system experiments for solving the 
#   minimum cut problem with graph stream 
# Note: Utilizes all-sketch subgraphs data structure
#

if [[ $# -lt 6 ]]; then
  echo "ERROR: Invalid Arguments!"
  echo "USAGE: exec_dir datasets_dir results_dir workers readers num_it"
  echo "----------------------------------------------------------------"
  echo "exec_dir:              Directory where the executable is placed"
  echo "datasets_dir:          Directory where datasets are placed"
  echo "results_dir:           Directory where csv file should be placed"
  echo "workers:               Number of graph workers"
  echo "readers:               Number of graph readers"
  echo "num_it:                Number of iterations"
  exit 1
fi

exec_dir=$1
datasets_dir=$2
result_dir=$3
workers=$4
readers=$5
num_it=$6

# Table of num_batch_per_buffer for each dataset
declare -A num_batch_table=(
  [kron_13_stream_binary]=54
  [kron_15_stream_binary]=216
  [kron_16_stream_binary]=540
  [kron_17_stream_binary]=540
  [ktree_13_2048_stream_binary_shuffled]=54
  [ktree_15_8192_stream_binary_shuffled]=216
  [ktree_16_16384_stream_binary_shuffled]=540
  [ktree_17_32768_stream_binary_shuffled]=540
  [ca_citeseer_stream_binary]=216
  [google_plus_stream_binary]=216
  [p2p_gnutella_stream_binary]=216
  [rec_amazon_stream_binary]=216
  [web_uk_stream_binary]=216
  [reddit_100active_13sub_corpus_stream_binary_shuffled_merged]=540)

# Datasets
kron_graphs=("kron_13_stream_binary" "kron_15_stream_binary"
             "kron_16_stream_binary" "kron_17_stream_binary")

ktree_graphs=("ktree_13_2048_stream_binary_shuffled" 
              "ktree_15_8192_stream_binary_shuffled"
              "ktree_16_16384_stream_binary_shuffled" 
              "ktree_17_32768_stream_binary_shuffled")

sparse_graphs=("ca_citeseer_stream_binary" "google_plus_stream_binary"
               "p2p_gnutella_stream_binary" "rec_amazon_stream_binary"
               "web_uk_stream_binary")

reddit_graphs=("reddit_100active_13sub_corpus_stream_binary_shuffled_merged")

out_file=runtime_results.csv

# Note Message for GPU performance
echo "Note: To ensure the best performance for GPU, make sure to maximize its clock speed by 'nvidia-smi -lgc [CLOCK_SPEED]'"

# Write header to outfile csv
echo "stream_file, num_batch_per_buffer, eps, ingestion_rate (1e6), memory_usage (MiB), query_latency (sec), approx_mc" > ${out_file}

eps_val="0.75"

# Execute benchmark
for stream_name in "${kron_graphs[@]}"
do
  for (( it=1; it <= $num_it; ++it ))
  do
    echo -n "$stream_name, ${num_batch_table[$stream_name]}, ${eps_val}, " >> ${out_file}
    ${exec_dir}/min_cut ${datasets_dir}/kron_connected/${stream_name} $workers $readers yes $eps_val false ${num_batch_table[$stream_name]}
  done
done

for stream_name in "${ktree_graphs[@]}"
do
  for (( it=1; it <= $num_it; ++it ))
  do
    echo -n "$stream_name, ${num_batch_table[$stream_name]}, ${eps_val}, " >> ${out_file}
    ${exec_dir}/min_cut ${datasets_dir}/ktree/${stream_name} $workers $readers yes $eps_val false ${num_batch_table[$stream_name]}
  done
done

for stream_name in "${sparse_graphs[@]}"
do
  for (( it=1; it <= $num_it; ++it ))
  do
    echo -n "$stream_name, ${num_batch_table[$stream_name]}, ${eps_val}, " >> ${out_file}
    ${exec_dir}/min_cut ${datasets_dir}/real_world/${stream_name} $workers $readers yes $eps_val false ${num_batch_table[$stream_name]}
  done
done

for stream_name in "${reddit_graphs[@]}"
do
  for (( it=1; it <= $num_it; ++it ))
  do
    echo -n "$stream_name, ${num_batch_table[$stream_name]}, ${eps_val}, " >> ${out_file}
    ${exec_dir}/min_cut ${datasets_dir}/reddit/${stream_name} $workers $readers yes $eps_val false ${num_batch_table[$stream_name]}
  done
done

mv build/$out_file $result_dir/mc_no_es.csv
