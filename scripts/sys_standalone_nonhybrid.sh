#
# Name: sys_standalone_nonhybrid.sh
#
# Measures standalone performance of each
#   subsystem in GPUSketch on kron graphs
#   for non-hybrid sketch algorithm.
#
# 1. CPU's vertex-based batcher 
# 2. CPU-GPU data transfer
# 3. GPU's sketch update
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

# Datasets
kron_graphs=("kron_13_stream_binary" "kron_15_stream_binary"
             "kron_16_stream_binary" "kron_17_stream_binary")

out_file=runtime_results.csv

# Note Message for GPU performance
echo "Note: To ensure the best performance for GPU, make sure to maximize its clock speed by 'nvidia-smi -lgc [CLOCK_SPEED]'"

# Write header to outfile csv
echo "stream_file, system, k, ingestion_rate (1e6)" > ${out_file}

kconn=("1" "2" "3" "4" "5")

# Execute benchmark
# 1. CPU's vertex-based batcher 
for stream_name in "${kron_graphs[@]}"
do
  for k in "${kconn[@]}"
  do
    for (( it=1; it <= $num_it; ++it ))
    do
        echo -n "$stream_name, CPU (Batcher), ${k}, " >> ${out_file}
        ${exec_dir}/st_cpu_batcher ${datasets_dir}/kron/${stream_name} $workers $readers ${k}
    done
  done
done

# 2. CPU-GPU data transfer
for stream_name in "${kron_graphs[@]}"
do
  for k in "${kconn[@]}"
  do
    for (( it=1; it <= $num_it; ++it ))
    do
        echo -n "$stream_name, CPU-GPU Transfer, ${k}, " >> ${out_file}
        ${exec_dir}/st_cpu_gpu_transfer ${datasets_dir}/kron/${stream_name} $workers $readers ${k} false
    done
  done
done

# 3. GPU's sketch update
for stream_name in "${kron_graphs[@]}"
do
  for k in "${kconn[@]}"
  do
    for (( it=1; it <= $num_it; ++it ))
    do
        echo -n "$stream_name, GPU Kernel, ${k}, " >> ${out_file}
         ${exec_dir}/st_gpu_sketch_update ${datasets_dir}/kron/${stream_name} $workers $readers ${k} false
    done
  done
done

mv ${out_file} $result_dir/sys_standalone_nonhybrid.csv