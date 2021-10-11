# %%
import numpy as np

X = np.array([
    'aa', 'ab', 'ac', 'ad', 'ba', 'bb', 'bc', 'bd', 'ca', 'cb', 'cc', 'cd',
    'da', 'db', 'dc', 'dd'
]).reshape(4, 4)
# %%
X


# %%
def shift(X):
    return np.roll(np.roll(X, -1, axis=0), -1, axis=1)


# %%
COV = X
for i in range(X.shape[0]):
    print(i)
    COV = shift(COV)
    print(COV)
# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats



means = np.array([[1, 1, 1]])
covs = np.array([[1, 0.9, 0.5], [0.9, 1, 0.1], [0.5, 0.1, 1]])
D = means.shape[1]
N = 100

stdevs = np.linalg.cholesky(covs)

true_sample = stdevs @ np.random.randn(D,N) + means.T

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(true_sample[0], true_sample[1], true_sample[2])

# %%
