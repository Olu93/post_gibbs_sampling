# %%
from IPython import get_ipython

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

cov_ab = 0.1
cov_ac = 0.9
cov_ad = 0.2
cov_bc = 0.1
cov_bd = 0.7
cov_cd = 0.8

means = np.array([[1, -2, 1, 3]]).T
covs = np.array([[3, cov_ab, cov_ac, cov_ad], [cov_ab, 2, cov_bc, cov_bd],
                 [cov_ac, cov_bc, 1, cov_cd], [cov_ad, cov_bd, cov_cd, 1]])
D = means.shape[0]
N = 300

stdevs = np.linalg.cholesky(covs)

true_sample = stdevs @ np.random.randn(D, N) + means

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(true_sample[0], true_sample[1], true_sample[2],
           c=true_sample[3])  # %%


# %%
DEBUG = True

def shift_covs(X, var_index):
    return np.roll(np.roll(X, -var_index, axis=0), -var_index, axis=1)


def shift_means(X, var_index):
    return np.roll(X, -var_index, axis=0)


def compute_conditional_probability(g, a, b, A, B, C):
    # a + C*B^{-1}(g - b)
    B_inv = np.linalg.inv(B)
    mu = a + C @ B_inv @ (g - b)
    # A - C * B^{-1} * C^T
    cov = A - C @ B_inv @ C
    return np.sqrt(cov) * np.random.randn(1) + mu


def gibbs_sample(joint_mu, joint_cov, sample_count, initial_sample=None):
    '''Does Gibbs sampling given the distribution's univariate conditionals.
    
    Returns a D x N matrix
    '''
    # collector = []
    
    D = len(joint_mu)

    # initializes an empty matrix for the samples
    samples = np.zeros((D, sample_count))

    # initialize the first sample to some arbitrary value
    
    samples[:, 0] = np.ones_like(samples[:, 0]) * 1 if initial_sample is None else initial_sample
    # samples[:, 0] = np.ones_like(samples[:, 0]) * 42
    for i in range(1, sample_count):
        # first set this sample equal to the previous sample
        prev_sample = samples[:, i - 1]
        next_sample = np.array(prev_sample)

        # now update the dimension whose turn it is using the conditional distribution
        # pass in all dimension from the previous sample except this dimension
        for d in range(D):
            shifted_joint_mu = shift_means(joint_mu, d)
            shifted_joint_cov = shift_covs(joint_cov, d)

            a = shifted_joint_mu[0]
            b = shifted_joint_mu[1:]

            A = shifted_joint_cov[0, 0]
            B = shifted_joint_cov[1:, 1:]
            C = shifted_joint_cov[1:, 0]
            print(next_sample)
            g = np.array([shift_means(next_sample, d)[1:]]).T
            next_sample[d] = compute_conditional_probability(g, a, b, A, B, C)

        # print()
        samples[:, i] = next_sample
    #     if DEBUG:
    #         collector.append(np.absolute(prev_sample-next_sample))
    # if DEBUG:
    #     print(np.array(collector).mean(axis=0))

    return samples


samples = gibbs_sample(means, covs, sample_count=100, initial_sample=true_sample[:, 42])
samples.T

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(true_sample[0], true_sample[1], true_sample[3],
           c=true_sample[2])  # %%
ax.set_xlim((-5,5))
ax.set_ylim((-5,5))
ax.set_zlim((-5,5))
# plt.clim((-5,5))
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(samples[0], samples[1], samples[3], c=samples[2])  # %%
ax.set_xlim((-5,5))
ax.set_ylim((-5,5))
ax.set_zlim((-5,5))
# plt.clim((-5,5))
fig.tight_layout()
plt.show()