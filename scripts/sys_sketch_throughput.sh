#
# Name: sys_sketch_throughput.sh
#
# Sketch update throughput benchmark on CPU and GPU
#

if [[ $# -lt 4 ]]; then
  echo "ERROR: Invalid Arguments!"
  echo "USAGE: exec_dir datasets_dir results_dir num_threads"
  echo "----------------------------------------------------------------"
  echo "exec_dir:              Directory where the executable is placed"
  echo "datasets_dir:          Directory where datasets are placed"
  echo "results_dir:           Directory where csv file should be placed"
  echo "num_threads:           Number of CPU threads performing sketch update"
  exit 1
fi

exec_dir=$1
datasets_dir=$2
result_dir=$3
num_threads=$4

# Number of nodes
num_nodes=("131072" "262144" "524288")

out_file=runtime_results.csv

# Note Message for GPU performance
echo "Note: To ensure the best performance for GPU, make sure to maximize its clock speed by 'nvidia-smi -lgc [CLOCK_SPEED]'"

# Write header to outfile csv
echo "num_nodes, system, num_threads (CPU), N (1e6), update_throughput (1e6)" > ${out_file}

# Execute benchmark
for num_node in "${num_nodes[@]}"
do
  ${exec_dir}/cpu_sketch_throughput_stream $num_node $num_threads
done

for num_node in "${num_nodes[@]}"
do
  ${exec_dir}/gpu_sketch_throughput_stream $num_node
done

mv ${out_file} $result_dir/cpu_gpu_sketch_throughput.csv