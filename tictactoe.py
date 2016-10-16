"""
Implementation of TicTacToe for reinforcement learning
The API is roughly equivalent to the OpenAI gym environments
"""

import numpy as np

class TicTacToe():

    def __init__(self, board_size=3, opponent='random'):
        """
        Create a board_size x board_size tic-tac-toe board
        
        opponent can be either 'random', 'human' or a function
        which takes the current state of the game as input and
        returns the index of the action to be played. 
        """
        self.board_size = board_size
        self.opponent = opponent
        self.reset()

        if opponent == 'random':
            self._do_opponent_move = _random_move
        elif opponent == 'human':
            self._do_opponent_move = _human_move
        else:
            self._do_opponent_move = opponent

    def reset(self):
        """
        Reset the game to its initial state (empty board)
        Return the empty board
        """
        board_size = self.board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int)
        self.done = False
        return self.board

    def render(self):
        """
        Write a human readable version of the game to the console
        """
        rows = [' | '.join(str(row)[1:-1].split()) for row in self.board]
        seperator = '-' * (self.board_size * 3 + self.board_size - 3)
        board = '\n{}\n'.format(seperator).join(rows)
        board = board.replace('-1', 'o').replace('1', 'x').replace('0', ' ')
        print(board)

    def step(self, action):
        """
        Perform the action and let the opponent play their action

        Return board, reward, done, None
        The last element of the tuple is always None, but allows
        for compliance with the OpenAI gym environment API
        """
        flat_board = self.board.ravel()
        
        # Check if the game is already over
        if self.done:
            return self.board, 0., self.done, None

        # Check for illegal move
        if flat_board[action] != 0:
            self.done = True
            return self.board, -2., self.done, None
        
        # Do the action and check if the game is over
        flat_board[action] = 1
        reward = self._check_if_over()
        if reward == 1: # Won
            self.done = True
            return self.board, 1., self.done, None
        if reward == 2: # Draw
            self.done = True
            return self.board, -0.1, self.done, None
        
        # Opponent action
        opp_action = self._do_opponent_move(self)
        if flat_board[opp_action] != 0:
            raise ValueError('Invalid move from opponent')
        else:
            flat_board[opp_action] = -1

        reward = self._check_if_over()
        if reward == -1: # Lost
            self.done = True
            return self.board, -1., self.done, None
        if reward == 2: # Draw
            self.done = True
            return self.board, -0.1, self.done, None
        
        # The game goes on
        return self.board, 0., self.done, None


    def _check_if_over(self):
        """
        Check if the game isn't over yet
        Return:
            0 if the game isn't over
            1 if the player has won
            2 if the game is a draw
            -1 if the player has lost
        """
        col = self.board.sum(axis=0)
        row = self.board.sum(axis=1)
        diag = self.board.trace()
        adiag = np.flipud(self.board).trace()
        sums = np.hstack((col, row, diag, adiag))
        if (sums == self.board_size).any():
            return 1
        elif (sums == -self.board_size).any():
            return -1
        elif (self.board != 0).all():
            return 2
        else:
            return 0

def _random_move(game):
    """
    Pick a move at random
    """
    flat_board = game.board.ravel()
    action = np.random.choice(np.where(flat_board == 0)[0])
    return action
    
def _human_move(game):
    """
    Render the board and ask the user for input
    """
    game.render()
    action = int(input('Action: '))
    return action
        

