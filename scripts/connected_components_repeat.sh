#
# Name: connected_components_repeat.sh
#
# Full system experiments for solving the 
#   connected components problem with repeated graph stream
#

if [[ $# -lt 5 ]]; then
  echo "ERROR: Invalid Arguments!"
  echo "USAGE: exec_dir datasets_dir results_dir workers readers"
  echo "----------------------------------------------------------------"
  echo "exec_dir:              Directory where the executable is placed"
  echo "datasets_dir:          Directory where datasets are placed"
  echo "results_dir:           Directory where csv file should be placed"
  echo "workers:               Number of graph workers"
  echo "readers:               Number of graph readers"
  exit 1
fi

exec_dir=$1
datasets_dir=$2
result_dir=$3
workers=$4
readers=$5

# Table of num_batch_per_buffer for each dataset
declare -A num_batch_table=(
  [kron_13_stream_binary]=54
  [kron_15_stream_binary]=216
  [kron_16_stream_binary]=540
  [kron_17_stream_binary]=540
  [ca_citeseer_stream_binary]=216
  [google_plus_stream_binary]=216
  [p2p_gnutella_stream_binary]=216
  [rec_amazon_stream_binary]=216
  [web_uk_stream_binary]=216)

# Datasets
kron_graphs=("kron_13_stream_binary" "kron_15_stream_binary"
             "kron_16_stream_binary" "kron_17_stream_binary")

sparse_graphs=("ca_citeseer_stream_binary" "google_plus_stream_binary"
               "p2p_gnutella_stream_binary" "rec_amazon_stream_binary"
               "web_uk_stream_binary")

out_file=runtime_results.csv

# Note Message for GPU performance
echo "Note: To ensure the best performance for GPU, make sure to maximize its clock speed by 'nvidia-smi -lgc [CLOCK_SPEED]'"

# Write header to outfile csv
echo "stream_file, num_batch_per_buffer, ingestion_rate (1e6), memory_usage (MiB), query_latency (sec)" > ${out_file}

# Execute benchmark
for stream_name in "${kron_graphs[@]}"
do
  echo -n "$stream_name, ${num_batch_table[$stream_name]}, " >> ${out_file}
  ${exec_dir}/cuda_process_stream_repeat ${datasets_dir}/kron/${stream_name} $workers $readers 11 ${num_batch_table[$stream_name]}
done

for stream_name in "${sparse_graphs[@]}"
do
  echo -n "$stream_name, ${num_batch_table[$stream_name]}, " >> ${out_file}
  ${exec_dir}/cuda_process_stream_repeat ${datasets_dir}/real_world/${stream_name} $workers $readers 11 ${num_batch_table[$stream_name]}
done

mv ${out_file} $result_dir/cc_repeat.csv
