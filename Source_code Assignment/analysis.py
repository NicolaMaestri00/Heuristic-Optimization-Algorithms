import numpy as np
import pandas as pd
import os
dfs = []
for file in os.listdir('output'):
    if 'grid_search' not in file:
        continue
    
    dfs.append(pd.read_csv('output/' + file))
    
df = pd.concat(dfs)
df = df.reset_index(drop=True)

objective_means = df.groupby(by='name').objective.mean()
print(objective_means)
df['gain'] = df.apply(lambda row: (row['objective'] - objective_means.loc[row['name']])/objective_means.loc[row['name']], axis=1)
print(df['gain'])

best = df.groupby(by=['temperature', 'alpha', 'gamma']).gain.agg(['mean', 'std']).reset_index()
print(best.columns)
print(best[best['mean'] == best['mean'].min()])

print(df.gain.mean(), df.gain.std(), df.shape, best[best['mean'] == best['mean'].min()]['mean'])
from scipy.stats import norm
df['group'] = (df.gamma==.7)# (df.temperature == 100)&(df.alpha == .92)&(df.gamma==.7)
mu0 = df.gain.mean()
sd0 = df.gain.std()
mu1 = df[df.group == True].gain.mean()
sd1 = df[df.group == True].gain.std()
# print('T-score:\t', (mu1-mu0)/sd0/np.sqrt(18*11))

print(mu0, mu1, sd0, sd1)

df['pdf0'] = norm.logpdf(df.gain, loc=mu0, scale=sd0)
likelihood0 = df.pdf0.sum()


df['pdf1'] = norm.logpdf(df.gain, loc=mu1, scale=sd1)
likelihood1 = df[df.group == True].pdf1.sum() + df[df.group==False].pdf0.sum()

print('Log Bayes Factor:\t',(likelihood1 - likelihood0)/np.log(10))
print('Bayes Factor:\t', np.exp(likelihood1 - likelihood0))

###
g0 = df[df.group == True].gain
g1 = df[df.group == False].gain

mu0 = g0.mean()
mu1 = g1.mean()
s0 = g0.var()
s1 = g1.var()
n0 = g0.shape[0]
n1 = g1.shape[0]

ttest = ( mu1 - mu0 ) / np.sqrt(1/n0 + 1/n1) / np.sqrt(((n0-1)*s0 + (n1-1)*s1) / (n1+n0-2))
print('T-score:\t', ttest)
print('DF:\t', n1+n0-2)

import seaborn as sns
import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# sns.boxenplot(data=df, y='temperature', x='gain', hue='alpha', orient='h').set(
#     title = 'Gain over temperature and cooling',
# )
# plt.show()

# plt.figure(figsize=(10, 5))
# sns.boxenplot(data=df, y='gamma', x='gain', orient='h').set(
#     title = 'Gain over ɣ',
# )
# plt.show()

# plt.figure(figsize=(10, 5))
# sns.scatterplot(
#     data = best,
#     y = 'temperature',
#     x = 'mean', 
#     hue = 'alpha', 
#     style='gamma',
#     size = 'std', 
#     sizes=(200,1000),
# ).set(
#     yscale='log',
#     title = 'Best triplet'
# )
# plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
# plt.show()