import pandas as pd
from astropy import table
from ekfparse import query

gz = table.Table.read(
    '../local_data/galaxy_zoo_classifications/dr5/gzdv5.dat',
    readme='../local_data/galaxy_zoo_classifications/dr5/ReadMe',
    format='cds'
)

gz_df = gz.to_pandas ()


mmatch, gzmatch = query.match_catalogs(catalog, gz_df, coordkeysB=['RAdeg','DEdeg'])

has_classification = gzmatch['NbDEO'] > 0
is_eod = (gzmatch['DEOyes'] > 0.5) & (gzmatch['EOBNo'] > 0.5)
not_eod = ~is_eod & has_classification
is_eod &= has_classification

df = pd.DataFrame(index=mmatch.index, columns=['classification'])
df.loc[:,'classification'] = 0
df.loc[is_eod.values, 'classification'] = 2
df.loc[not_eod.values, 'classification'] = 1

df.to_csv('../local_data/galaxy_zoo_classifications/classifications.csv')