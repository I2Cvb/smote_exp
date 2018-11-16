ma = cv_results.groupby('smote__sampling_strategy')[['test_score']].mean()
mstd = cv_results.groupby('smote__sampling_strategy')[['test_score']].std()

plt.plot(ma.index, ma)
plt.fill_between(mstd.index,
                 ma.values.ravel() - mstd.values.ravel(),
                 ma.values.ravel() + mstd.values.ravel(),
                 alpha=0.2)
