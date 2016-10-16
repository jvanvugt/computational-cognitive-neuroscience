from policy_gradient import PolicyGradientNet
from tictactoe import TicTacToe

net = PolicyGradientNet(learning_rate=1e-2)
net.train(TicTacToe())