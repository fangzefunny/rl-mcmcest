import os 
import pickle
import numpy as np
import jax.numpy as jnp
import pandas as pd 

import time 
import jax
from jax import vmap
import numpyro as npyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from numpyro.infer.reparam import TransformReparam
from numpyro.contrib.control_flow import scan 


from scipy.stats import gamma, beta 
from scipy.special import softmax

# ------------------------------#
#      Simulated Experiment     #
#-------------------------------#

class rl:
    name    = 'rlQ'
    priors  = [gamma(a=1, scale=5), beta(a=1.2, b=1.2)]
    bnds    = [(0, 50), (0, 1)]
    pbnds   = [(0, 10), (0,.5)]
    n_param = len(bnds)

    def __init__(self, nS, nA):
        self.nS   = nS
        self.nA   = nA
        
    def sim(self, params, T=320, seed=42):
 
        self.q = np.zeros([self.nS, self.nA])
        data = {'s': [], 'a': [], 'a_cor':[], 'r': [], 't': []}
        rng  = np.random.RandomState(seed)

        # decompose parameters
        self.alpha = params[0]
        self.beta = params[1]
        rew_fn = np.array([[.9, .1], [.1, .9]])

        for t in range(T):

            # generate a trajectory 
            s = rng.choice(self.nS) 
            p = softmax(self.beta*self.q[s, :])
            c = rng.choice(self.nA, p=p)
            a_cor = rng.choice(self.nA, p=rew_fn[s, :])
            r = 1*(c==a_cor)
            self.q[s, c] += self.alpha*(r-self.q[s, c])

            # save data 
            data['s'].append(s)
            data['a'].append(c)
            data['a_cor'].append(a_cor)
            data['r'].append(r)
            data['t'].append(t)
        
        return pd.DataFrame.from_dict(data)
    
    def vmap_model(self, data):

        def q_update(q, info):

            # upack 
            s, a, r, alpha, beta = info

            # forward  
            f = beta*q[s, :]
            p = jnp.exp(f - jnp.log(jnp.exp(f).sum()))[1]

            # update 
            rpe = r - q[s, a]
            q = q.at[s, a].set(q[s, a]+alpha*rpe)

            return q, p
        
        def run_subj(alpha0, beta0, s, a, r):
            q0 = jnp.zeros([self.nS, self.nA])
            T = len(r)
            alpha = alpha0 * jnp.ones([T,]) 
            beta  = beta0 * jnp.ones([T,]) 
            q, probs = scan(q_update, q0, (s, a, r, alpha, beta), length=T)
            return probs
        
        # get subject list 
        sub_lst = data['sub_id'].unique()
        n_sub  = len(sub_lst)
        s = jnp.vstack([data.query(f'sub_id=={sub_id}')['s'].values for sub_id in sub_lst])
        a = jnp.vstack([data.query(f'sub_id=={sub_id}')['a'].values for sub_id in sub_lst])
        r = jnp.vstack([data.query(f'sub_id=={sub_id}')['r'].values for sub_id in sub_lst])
        a_mu   = npyro.sample(f'alpha_mu', dist.Normal(.4, .5))
        a_sig  = npyro.sample(f'alpha_sig', dist.HalfNormal(.5))
        b_mu   = npyro.sample(f'beta_mu', dist.Normal(1, .5))
        b_sig  = npyro.sample(f'beta_sig', dist.HalfNormal(.5))
        with npyro.plate('sub_id', n_sub):
            with npyro.handlers.reparam(config={'alpha': TransformReparam(), 
                                                'beta': TransformReparam()}):
                alpha = npyro.sample(
                    'alpha', dist.TransformedDistribution(dist.Normal(0., 1.),
                             dist.transforms.AffineTransform(a_mu, a_sig)))
                beta  = npyro.sample(
                    'beta', dist.TransformedDistribution(dist.Normal(0., 1.),
                             dist.transforms.AffineTransform(b_mu, b_sig)))

        probs = vmap(run_subj)(alpha, beta, s, a, r)
        npyro.sample(f'a_hat', dist.Bernoulli(probs=probs), obs=a)

    def loop_model(self, data):

        def q_update(q, info):

            # upack 
            s, a, r, alpha, beta = info

            # forward  
            f = beta*q[s, :]
            p = jnp.exp(f - jnp.log(jnp.exp(f).sum()))[1]

            # update 
            rpe = r - q[s, a]
            q = q.at[s, a].set(q[s, a]+alpha*rpe)

            return q, p
        
        # get subject list 
        sub_lst = data['sub_id'].unique()
        n_sub  = len(sub_lst)
        a_mu   = npyro.sample(f'alpha_mu', dist.Normal(.4, .5))
        a_sig  = npyro.sample(f'alpha_sig', dist.HalfNormal(.5))
        b_mu   = npyro.sample(f'beta_mu', dist.Normal(1, .5))
        b_sig  = npyro.sample(f'beta_sig', dist.HalfNormal(.5))
        with npyro.plate('sub_id', n_sub):
            with npyro.handlers.reparam(config={'alpha': TransformReparam(), 
                                                'beta': TransformReparam()}):
                alpha0 = npyro.sample(
                    'alpha', dist.TransformedDistribution(dist.Normal(0., 1.),
                             dist.transforms.AffineTransform(a_mu, a_sig)))
                beta0  = npyro.sample(
                    'beta', dist.TransformedDistribution(dist.Normal(0., 1.),
                             dist.transforms.AffineTransform(b_mu, b_sig)))
        # input 
        for i, sub_id in enumerate(sub_lst):
            q0 = jnp.zeros([self.nS, self.nA])
            s = data.query(f'sub_id=={sub_id}')['s'].values
            a = data.query(f'sub_id=={sub_id}')['a'].values
            r = data.query(f'sub_id=={sub_id}')['r'].values
            T = len(r)
            alpha = alpha0[i] * jnp.ones([T,]) 
            beta  = beta0[i] * jnp.ones([T,]) 
            q, probs = scan(q_update, q0, (s, a, r, alpha, beta), length=T)
            npyro.sample(f'a_hat_{sub_id}', dist.Bernoulli(probs=probs), obs=a)


    def sample(self, data, mode='loop', seed=1234, 
                    n_samples=50000, n_warmup=100000):

        # set the random key 
        rng_key = jax.random.PRNGKey(seed)

        # sampling 
        start_time = time.time()
        kernel = NUTS(eval(f'self.{mode}_model'))
        posterior = MCMC(kernel, num_chains=4,
                                 num_samples=n_samples,
                                 num_warmup=n_warmup)
        posterior.run(rng_key, data)
        samples = posterior.get_samples()
        posterior.print_summary()
        end_time = time.time()
        print(f'Sampling takes {end_time - start_time:2f}s')

        if not os.path.exists(d): os.mkdir(d)
        with open(f'data/rlq_{mode}.pkl', 'wb')as handle:
            pickle.dump(samples, handle)

if __name__ == '__main__':

    # generate data 
    opt_params = [[.25, 3], [.1, 1], [.7, 2]]
    sim_data = []

    for sub_id, opt_param in enumerate(opt_params):
        sim_datum = rl(2, 2).sim(opt_param)
        sim_datum['sub_id'] = sub_id
        sim_data.append(sim_datum)

    sim_data = pd.concat(sim_data, axis=0)

    # start sampling
    npyro.set_host_device_count(4)
    agent = rl(2, 2)
    agent.sample(sim_data, mode='loop')