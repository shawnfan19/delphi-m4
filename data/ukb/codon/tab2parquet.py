import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

input_file = "/nfs/research/birney/controlled_access/ukb-cnv/data_fetch/baskets/2017133/ukb676101.tab"
output_file = "/hps/nobackup/birney/users/sfan/delphi-data/ukb/tab.parquet"

chunksize = 10000
writer = None

for chunk in tqdm(
    pd.read_csv(input_file, sep="\t", dtype=str, na_filter=False, chunksize=chunksize)
):
    table = pa.Table.from_pandas(chunk, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(output_file, table.schema, compression="snappy")

    writer.write_table(table)

if writer is not None:
    writer.close()
