import pickle

import numpy as np

from tictactoe import TicTacToe
from policy_gradient import PolicyGradientNet

model = pickle.load(open('save.p', 'rb'))
pgnet = PolicyGradientNet(model=model)

game = TicTacToe(opponent='human')
obs = game.reset()
while not game.done:
    x = obs.ravel()
    probs, _ = pgnet.forward(x)
    action = int(np.random.choice(9, p=probs))
    obs, _, _, _ = game.step(action)
    game.render()

game.render()
