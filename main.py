import pickle
import numpy as np
import jax.numpy as jnp
import pandas as pd 

import time 
import jax
import numpyro as npyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
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
    
    def scan_model(self, data):

        def q_update(q, sar):

            # upack 
            s, a, r, alpha, beta = sar

            # forward  
            f = beta*q[s, :]
            p = jnp.exp(f - jnp.log(jnp.exp(f).sum()))[1]

            # update 
            rpe = r - q[s, a]
            q = q.at[s, a].set(q[s, a]+alpha*rpe)

            return q, p
        
        # get subject list 
        sub_lst = data['sub_id'].unique()
        
        # input 
        for sub_id in sub_lst:
            alpha0 = npyro.sample(f'alpha_{sub_id}', dist.Normal(.4, .5))
            beta0  = npyro.sample(f'beta_{sub_id}',  dist.Normal( 1, .5))
            q0 = jnp.zeros([self.nS, self.nA])
            s = data.query(f'sub_id=={sub_id}')['s'].values
            a = data.query(f'sub_id=={sub_id}')['a'].values
            r = data.query(f'sub_id=={sub_id}')['r'].values
            T = len(r)
            alpha = alpha0 * jnp.ones([T,]) 
            beta  = beta0 * jnp.ones([T,]) 
            q, probs = scan(q_update, q0, (s, a, r, alpha, beta), length=T)
            npyro.sample(f'a_hat_{sub_id}', dist.Bernoulli(probs=probs), obs=a)

    def loop_model(self, data):

        # load parameter 
        alpha = npyro.sample('alpha', dist.Normal(.4, .5))
        beta  = npyro.sample('beta',  dist.Normal( 1, .5))

        # init 
        q = jnp.zeros([self.nS, self.nA])
        probs = jnp.zeros([data.shape[0]])

        for t, row in data.iterrows():

            # forward
            s = row['s']
            f = beta*q[s, :]
            p = jnp.exp(f - jnp.log(jnp.exp(f).sum()))[1]
            probs = probs.at[t].set(p)

            # backward
            a = row['a']
            r = row['r']
            rpe = r - q[s, a]
            q = q.at[s, a].set(q[s, a] + alpha*rpe)
        
        npyro.sample('pi', dist.Bernoulli(probs=probs), 
                     obs=data['a'].values)


    def sample(self, data, mode='scan', seed=1234, 
                    n_samples=20000, n_warmup=50000):

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
    
        with open(f'data/rlq_{mode}.pkl', 'wb')as handle:
            pickle.dump(samples, handle)

if __name__ == '__main__':

    # generate data 
    opt_params = [[.25, 3], [.1, 1], [1, 2]]
    sim_data = []

    for sub_id, opt_param in enumerate(opt_params):
        sim_datum = rl(2, 2).sim(opt_param)
        sim_datum['sub_id'] = sub_id
        sim_data.append(sim_datum)

    sim_data = pd.concat(sim_data, axis=0)

    # start sampling
    npyro.set_host_device_count(4)
    agent = rl(2, 2)
    agent.sample(sim_data, mode='scan')