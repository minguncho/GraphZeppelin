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

# Change column names
df = df.rename(columns={'stream_file': 'Dataset',
                        'system': 'System',
                        'ingestion_rate (1e6)': 'Rate'})

# Change the data type for values
df['Rate'] = df['Rate'].astype(float)

# Remove '_stream_binary' in dataset values
df['Dataset'] = df['Dataset'].str.replace('_stream_binary', '', regex=False)

# Compute average
df = df.groupby(['Dataset', 'System', 'k'], as_index=False)['Rate'].mean()

# Custom Color scale
system_domain = ['CPU (Batcher)', 'CPU (Adj. List)', 'CPU-GPU Transfer', 'GPU Kernel']
system_range = ["#E66101", "#FDB863", "#B2ABD2", "#5E3C99"]

system_color_scale = alt.Scale(domain=system_domain, range=system_range)
system_shape_scale = alt.Scale(domain=system_domain)

shared_legend = alt.Legend(
  title="System", 
  symbolOpacity=1,
  values=['CPU (Batcher)', 'CPU-GPU Transfer', 'GPU Kernel'])

chart = alt.Chart(df).mark_line(
  strokeWidth=3, 
  opacity=0.5, 
  point=alt.OverlayMarkDef(size=60, filled=True)
  ).encode(
    x=alt.X('k:O', axis=alt.Axis(labelAngle=0), title='\u03ba'),
    y=alt.Y('Rate:Q', title='Rate (# of million edges/sec)'),
    shape=alt.Shape('System:N', legend=shared_legend).scale(system_shape_scale),
    color=alt.Color('System:N', legend=shared_legend).scale(system_color_scale),
    detail='System:N',
    column=alt.Column("Dataset:N", title='')
  ).properties(
      width=150,
      height=200
  ).configure_header(
    labelFontSize=14,
    labelPadding=10
  ).configure_axis(
    titleFontSize=14,
    labelFontSize=14
  ).configure_axisX(
    titleFont='serif', 
    titleFontStyle='italic',
  ).configure_legend(
    orient='none',    
    direction='horizontal',
    legendX=150,
    legendY=240,
    titleAlign='center',
    titleAnchor='middle',
    labelFontSize=14,
    titleFontSize=16
  )

chart.save("standalone_nonhybrid_perf.pdf")