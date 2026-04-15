import sys
import os
import altair as alt
import pandas as pd
import numpy as np

if len(sys.argv) != 2:
    print("Usage: <RESULT_FILE_PATH>")
    sys.exit(1)

result_file_path = sys.argv[1]
result_file_name = os.path.basename(result_file_path)

# Read result file
df = pd.read_csv(result_file_path)

# Clean the column names and values
df.columns = df.columns.str.strip()
df = df.astype(str).map(lambda x: x.strip())

# Drop columns that are not needed for the plot
df = df.drop('num_batch_per_buffer', axis=1)
df = df.drop('ingestion_rate (1e6)', axis=1)
df = df.drop('query_latency (sec)', axis=1)

# Change the data type for values
df['eps'] = df['eps'].astype(float)
df['approx_mc'] = df['approx_mc'].astype(float)
df['memory_usage (MiB)'] = df['memory_usage (MiB)'].astype(float)

# Covert memory usage to GiB
df = df.rename(columns={'memory_usage (MiB)': 'memory_usage (GiB)'})
df['memory_usage (GiB)'] = (df['memory_usage (GiB)'] / 1024)
df['memory_usage (GiB)'] = df.groupby(['stream_file', 'eps'])['memory_usage (GiB)'].transform('mean').round(2)

############################
# Generate plot for kron_17
############################
exact_mincut = 500

kron17_df = df[df['stream_file'].str.contains('kron_17_stream_binary')].copy()
kron17_df = kron17_df.assign(Color=1)
kron17_df = kron17_df.assign(Minimum=(1-kron17_df['eps'])*exact_mincut)
kron17_df = kron17_df.assign(MinimumValue=100)
kron17_df = kron17_df.assign(Range=(kron17_df['eps'])*exact_mincut)
kron17_df = kron17_df.assign(Exact=exact_mincut)

exact = alt.Chart(kron17_df).mark_rule(color='green').encode(
  x = 'Exact:Q'
).properties(width=350, height=150)

full_range_chart = alt.Chart(kron17_df).mark_rule(opacity=0.03, color='green').encode(
  x = alt.X('Minimum:Q'),
  x2 = alt.X2('Exact:Q'),
  y = alt.Y("eps:N", axis=alt.Axis(title='ε')).scale(reverse=True),
  size = alt.value(15)
).properties(width=350, height=150)

full_range_chart2 = alt.Chart(kron17_df).mark_boxplot(extent="min-max", color="black").encode(
  alt.X("approx_mc:Q", title='Approximated Minimum Cut Values', scale=alt.Scale(domainMax=525)),
  alt.Y("memory_usage (GiB):N", axis=alt.Axis(title='Peak Memory Usage (GiB)')),
).properties(width=350, height=150)

range_result = (full_range_chart + full_range_chart2).resolve_scale(y='independent')
range_result += exact

range_result.save("kron17_mc_approx_boxplot.pdf")

#############################
# Generate plot for ktree_17
#############################
exact_mincut = 32768

ktree17_df = df[df['stream_file'].str.contains('ktree_17_32768_stream_binary_shuffled')].copy()
ktree17_df = ktree17_df.assign(Color=1)
ktree17_df = ktree17_df.assign(Minimum=(1-ktree17_df['eps'])*exact_mincut)
ktree17_df = ktree17_df.assign(MinimumValue=27000)
ktree17_df = ktree17_df.assign(Range=(ktree17_df['eps'])*exact_mincut)
ktree17_df = ktree17_df.assign(Exact=exact_mincut)

exact = alt.Chart(ktree17_df).mark_rule(color='green').encode(
  x = 'Exact:Q'
).properties(width=350, height=150)

full_range_chart = alt.Chart(ktree17_df).mark_rule(opacity=0.03, color='green').encode(
  x = alt.X('Minimum:Q'),
  x2 = alt.X2('Exact:Q'),
  y = alt.Y("eps:N", axis=alt.Axis(title='ε')).scale(reverse=True),
  size = alt.value(15)
).properties(width=350, height=150)

full_range_chart2 = alt.Chart(ktree17_df).mark_boxplot(extent="min-max", color='black').encode(
  alt.X("approx_mc:Q", title='Approximated Minimum Cut Values', axis=alt.Axis(values=[0, 10000, 20000, 30000])),
  alt.Y("memory_usage (GiB):N", axis=alt.Axis(title='Peak Memory Usage (GiB)')),
).properties(width=350, height=150)

range_result = (full_range_chart + full_range_chart2).resolve_scale(y='independent')
range_result += exact

range_result.save("ktree17_mc_approx_boxplot.pdf")