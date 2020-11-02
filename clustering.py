import pandas
df = pandas.read_csv("AirlinesCluster.csv")
print(df)
print(df.shape)

# filling empties, encoding is needed where possible
subset = df[['FlightMiles','FlightTrans','DaysSinceEnroll']]

array = subset.values
X = array[:,0:3] # no response y, or label thus it is unsupervised

# train
from sklearn.cluster import KMeans
model = KMeans(n_clusters=4, random_state=42)
model.fit(X)

print('Learning// Done.')

# get the means for each cluster/per 3 columns
centronoids = model.cluster_centers_

dataframe = pandas.DataFrame(centronoids, columns=['FlightMiles','FlightTrans','DaysSinceEnroll'])

print(dataframe)

# who is in this clusters?
subset['label'] = model.labels_
subset = subset[subset['label'] == 2]
print(subset)

import xlwt
subset.to_excel('cluster3.xls', columns=['FlightMiles', 'FlightTrans', 'DaysSinceEnroll'])