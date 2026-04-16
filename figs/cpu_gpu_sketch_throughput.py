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
df = df.drop('num_threads (CPU)', axis=1)

# Change column names
df = df.rename(columns={'num_nodes': '|V|',
                        'system': 'System',
                        'N (1e6)': 'N',
                        'update_throughput (1e6)': 'Rate'})

# Change the data type for values
df['|V|'] = df['|V|'].astype(float)
df['N'] = df['N'].astype(float)
df['Rate'] = df['Rate'].astype(float)

# Convert each num_nodes values to powers of 2
df['|V|'] = df['|V|'].apply(lambda x: f"2^{int(np.log2(x))}")

# Custom Color scale
colorScale = alt.Scale(range=["#E66101", "#FDB863", "#5E3C99"])

# CPU df
cpu_df = df[df['System'] == 'CPU'].copy(deep=False)

# Plot cpu_data
cpu_chart = alt.Chart(cpu_df).mark_line().encode(
  x=alt.X('N:Q', axis=alt.Axis(labelAngle=0), title='Stream Length (E+06)').scale(type="log", domainMax=5000),
  y=alt.Y('Rate:Q', title='Rate (# of million edges per second)').scale(type="pow"),
  color=alt.Color('|V|:N').scale(colorScale),
  strokeDash="System"
).properties(width=200, height=175)

# GPU df
gpu_df = df[df['System'] == 'GPU'].copy(deep=False)

# Plot gpu_data
gpu_chart = alt.Chart(gpu_df).mark_line().encode(
  x=alt.X('N:Q', axis=alt.Axis(labelAngle=0), title='Stream Length (E+06)').scale(type="log", domainMax=5000),
  y=alt.Y('Rate:Q', title='Rate (# of million edges per second)'),
  color=alt.Color('|V|:N').scale(colorScale),
  strokeDash="System"
).properties(width=200, height=175)

# Compute Improvements
improve_data = gpu_df.copy()
improve_data['Rate'] = gpu_df['Rate'].values / cpu_df['Rate'].values

improve_chart = alt.Chart(improve_data).mark_line().encode(
  x=alt.X('N:Q', axis=alt.Axis(labelAngle=0), title='Stream Length (E+06)').scale(type="log", domainMax=5000),
  y=alt.Y('Rate:Q', title='Improvement').axis(values=[0, 10, 20]),
  color=alt.Color('|V|:N').scale(colorScale),
).properties(width=200, height=75) 

chart = cpu_chart + gpu_chart & improve_chart
chart.save(result_file_name[:-4] + ".pdf")