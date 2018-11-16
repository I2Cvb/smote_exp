import os
import pandas as pd
import matplotlib.pyplot as plt

cv_results = pd.read_csv(
    os.path.join('results', 'diabetes.csv'), index_col=0)

ma = cv_results.groupby('smote__sampling_strategy')[['test_score']].mean()
mstd = cv_results.groupby('smote__sampling_strategy')[['test_score']].std()

plt.plot(ma.index, ma)
plt.fill_between(mstd.index,
                 ma.values.ravel() - mstd.values.ravel(),
                 ma.values.ravel() + mstd.values.ravel(),
                 alpha=0.2)
plt.show()
