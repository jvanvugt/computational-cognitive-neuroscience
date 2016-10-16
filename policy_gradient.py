"""
Policiy gradient network.
Largely based on the excellent blogpost by Andrej Karpathy
http://karpathy.github.io/2016/05/31/rl/
"""

import time
import pickle
import numpy as np

class PolicyGradientNet():

    def __init__(self, model=None, D=9, H=9, A=9, batch_size=9, 
                       learning_rate=1e-3, gamma=0.99, decay_rate=0.99):
        """
        Model can be a dictionary with attributes W1 and W2 to resume
        from a checkpoint.
        Otherwise, a new model will be constructed with input D, H hidden
        units and A outputs.
        """
        if model is not None:
            self.model = model
        else:
            self.model = {}
            # Xavier initialization
            self.model['W1'] = np.random.randn(D, H) / np.sqrt(D)
            self.model['W2'] = np.random.randn(H, A) / np.sqrt(H)
        self.A = A
        # update buffers that add up gradients over a batch
        self.grad_buffer = {k:np.zeros_like(v) for k,v in self.model.items()} 
        # rmsprop memory
        self.rms_cache = {k:np.zeros_like(v) for k,v in self.model.items()}
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.decay_rate = decay_rate

    def forward(self, x):
        """
        Forward pass.
        x is the observation
        Return the probabilities for each action and hidden state h
        """
        h = np.dot(x, self.model['W1'])
        h[h<0] = 0 # ReLU nonlinearity
        logp = np.dot(h, self.model['W2'])
        p = self._softmax(logp)
        return p, h

    def backward(self, epx, eph, epdlogp):
        """
        backward pass. 
        (eph is array of intermediate hidden states)
        Return the gradients for the weights
        """
        dW2 = np.dot(eph.T, epdlogp)
        dh = np.dot(epdlogp, self.model['W2'].T)
        dh[eph<=0] = 0 # backprop relu
        dW1 = np.dot(epx.T, dh)
        return {'W1':dW1, 'W2':dW2}

    def train(self, env, render=False):
        observation = env.reset()
        xs,hs,dlogps,drs = [],[],[],[]
        running_reward = None
        reward_sum = 0
        episode_number = 0
        while True:
            if render: env.render()

            x = observation.ravel()

            # forward the policy network 
            # and sample an action from the returned probability
            aprob, h = self.forward(x)
            action = int(np.random.choice(self.A, p=aprob)) # roll the dice!

            # record various intermediates (needed later for backprop)
            xs.append(x) # observation
            hs.append(h) # hidden state
            y = np.zeros(self.A) # a "fake label"
            y[action] = 1
            # grad that encourages the action that was taken to be taken
            dlogps.append(y - aprob)
            # step the environment and get new measurements
            observation, reward, done, _ = env.step(action)
            reward_sum += reward
            
            # record reward (has to be done after 
            # we call step() to get reward for previous action)
            drs.append(reward) 

            if done: # an episode finished
                episode_number += 1

                # stack together all inputs, hidden states, 
                # action gradients, and rewards for this episode
                epx = np.vstack(xs)
                eph = np.vstack(hs)
                epdlogp = np.vstack(dlogps)
                epr = np.vstack(drs)
                xs,hs,dlogps,drs = [],[],[],[] # reset array memory

                # compute the discounted reward backwards through time
                discounted_epr = self._discount_rewards(epr)
                # standardize the rewards to be unit normal 
                # (helps control the gradient estimator variance)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)

                # modulate the gradient with advantage 
                # (PG magic happens right here.)
                epdlogp *= discounted_epr
                grad = self.backward(epx, eph, epdlogp)
                # accumulate grad over batch
                for k in self.model: self.grad_buffer[k] += grad[k] 

                # perform rmsprop parameter update every batch_size episodes
                if episode_number % self.batch_size == 0:
                    for k,v in self.model.items():
                        g = self.grad_buffer[k] # gradient
                        self.rms_cache[k] = (self.decay_rate*self.rms_cache[k] +
                                             (1-self.decay_rate) * g**2)
                        self.model[k] += (self.learning_rate * g / 
                                          (np.sqrt(self.rms_cache[k])+1e-5))
                        # reset batch gradient buffer
                        self.grad_buffer[k] = np.zeros_like(v) 

                # boring book-keeping
                running_reward = (reward_sum if running_reward is None else 
                                 running_reward * 0.99 + reward_sum * 0.01)
                print('Episode %d reward total was %f. running mean: %f' % 
                            (episode_number, reward_sum, running_reward))
                if episode_number % 100 == 0:
                    # Save the model
                    did_save = False
                    while not did_save:
                        try:
                            with open('save.p', 'wb') as f: 
                                pickle.dump(self.model, f)
                                did_save = True
                        except OSError:
                            time.sleep(1)
                
                reward_sum = 0
                observation = env.reset() # reset env
                prev_x = None

    def _softmax(self, x):
        """
        Softmax function to normalize logprobs
        """
        exps = np.exp(x).sum() 
        return np.exp(x) / exps

    def _discount_rewards(self, r):
        """
        Take 1D float array of rewards and 
        compute discounted reward
        """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r