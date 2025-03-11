if [[ $# -lt 4 ]]; then
  echo "ERROR: Invalid Arguments!"
  echo "USAGE: results_dir workers readers stream_files[+]"
  echo "results_dir:  Directory where csv file should be placed"
  echo "workers:      Number of graph workers"
  echo "readers:      Number of graph readers"
  echo "stream_files: One or more stream files to process"
  exit 1
fi


result_dir=$1
workers=$2
readers=$3
shift 3

cd build

out_file=runtime_results.csv

# Write header to outfile csv
echo "stream_file, ingestion_rate, memory_usage, query_latency" > $out_file

# Process input files
for input in $@
do
  echo "============================================================"
  echo "============================================================"

  stream_name=`basename $input`
  echo -n "$stream_name, " >> $out_file
  ./min_cut $input $workers $readers yes
done

cd -
mv build/$out_file $result_dir/mc_no_edge_store.csv
