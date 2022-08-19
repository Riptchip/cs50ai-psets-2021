"""
Tic Tac Toe Player
"""

import math
from random import randint

X = "X"
O = "O"
EMPTY = None


# Returns starting state of the board.
def initial_state():
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


# Returns player who has the next turn on a board.
def player(board):
    x = 0

    for i in range(3):
        for j in range(3):
            if board[i][j] != EMPTY:
                x += 1
    
    if x % 2 == 1:
        return O

    return X


# Returns set of all possible actions (i, j) available on the board.
def actions(board):
    possibleActions = []
    
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possibleActions.append((i, j))

    return possibleActions


# Returns the board that results from making move (i, j) on the board.
def result(board, action):
    x = action[0]
    y = action[1]

    # Check if the given action is a valid one
    if board[x][y] != EMPTY or x > 2 or y > 2:
        raise Exception("Invalid action")

    # Make a copy of the original board
    newBoard = initial_state()

    for i in range(3):
        for j in range(3):
            newBoard[i][j] = board[i][j]

    # Make action on the new board and return
    newBoard[x][y] = player(board)

    return newBoard


# Returns the winner of the game, if there is one.
def winner(board):
    # Check diagonals
    if board[0][0] == board[1][1] and board[1][1] == board[2][2] and board[0][0] != EMPTY:
        return board[0][0]
    elif board[0][2] == board[1][1] and board[1][1] == board[2][0] and board[0][2] != EMPTY:
        return board[0][2]
    
    # Check horizontals
    for i in range(3):
        if board[i][0] == board[i][1] and board[i][1] == board[i][2] and board[i][0] != EMPTY:
            return board[i][0]
    
    # Check verticals
    for i in range(3):
        if board[0][i] == board[1][i] and board[1][i] == board[2][i] and board[0][i] != EMPTY:
            return board[0][i]

    return None


# Returns True if game is over, False otherwise.
def terminal(board):
    if winner(board) != None or actions(board) == []:
        return True

    return False


# Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
def utility(board):
    if winner(board) != None:
        return 1 if winner(board) == X else -1
    
    return 0


# Returns the optimal action for the current player on the board.
def minimax(board):
    if terminal(board):
        return None

    # Get possible moves and set util variables
    frontier = actions(board)
    bestWorld = 1 if player(board) == X else -1
    tieMoves = []

    # Check for the optimal(s) action(s)
    for i in range(len(frontier)):
        move = frontier[i]
        
        score = calculate(result(board, move))

        # If have one that lead to the victory play this immediatly else add to the tie moves since AI don't do lost moves
        if score == bestWorld:
            return move
        elif score == 0:
            tieMoves.append(move)

    return tieMoves[randint(0, len(tieMoves) - 1)]


# Returns the better score that the player can get on the given board
def calculate(board):
    if terminal(board):
        return utility(board)
    
    # Get possible moves and set util variables
    frontier = actions(board)
    bestScore = 1 if player(board) == X else -1
    tieMove = False

    # Check the scores that each action can bring
    for i in range(len(frontier)):
        score = calculate(result(board, frontier[i]))

        if score == bestScore:
            return bestScore
        if score == 0:
            tieMove = True

    # If there's no action that leads to the best score return 0 (tie score) if have a move that leads to a tie or return the worst score
    return 0 if tieMove else -bestScore
