import numpy as np
import matplotlib.pyplot as plt
from Gridworld import  Gridworld , printPolicy , printV
from tabulate import tabulate
from policyEvaluation import evaluatePolicy


def iterateValues(grid, V, policy, GAMMA, THETA):
    converged = False
    i = 0
    while not converged:
        DELTA = 0
        for state in grid.statespace:
            i += 1
            oldV = V[state]
            newV = []
            for action in grid.actionspace:
                for key in grid.p[(state,action)]:
                    (tp,newState,reward,oldState,act) = key
                    #if state == oldState and action == act:
                    newV.append(tp*(reward+GAMMA*V[newState]))
            newV = np.array(newV)
            bestV = np.where(newV == newV.max())[0]
            bestState = np.random.choice(bestV)
            V[state] = newV[bestState]
            DELTA = max(DELTA, np.abs(oldV-V[state]))
            converged = True if DELTA < THETA else False

    for state in grid.statespace:
        newValues = []
        actions = []
        i += 1
        for action in grid.actionspace:
            for key in grid.p[(state,action)]:
                (tp,newState,reward,oldState,act) = key
                #if state == oldState and action == act:
                newValues.append(tp*(reward+GAMMA*V[newState]))
            actions.append(action)
        newValues = np.array(newValues)
        bestActionIDX = np.where(newValues == newValues.max())[0]
        bestActions = actions[bestActionIDX[0]]
        policy[state] = bestActions
    print(i, 'sweeps of state space for value iteration')
    return V, policy

if __name__ == '__main__':
    # map magic squares to their connecting square
    wall=[(12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13), (12, 14), (7, 2), (8, 2), (9, 2), (10, 2), (11, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10)]
    #wall=[(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10)]
    env = Gridworld(15, 15, wall,0.2)
    # print(env.getAgentRowAndColoumn())
    # env.setState(1)
    # print(env.getAgentRowAndColoumn())
    env.renderHeatMap()
    Gamma = 1
    Theta = 1e-11
    stable = False

    # initialize V(s)
    Z = {}
    for state in env.statespaceplus:
        Z[state] = 0

    # Reinitialize policy
    policy_2 = {}
    for state in env.statespace:
        policy_2[state] = [key for key in env.possibleaction]


    # 2 round of value iteration ftw
    for i in range(2):
        Z, policy_2 = iterateValues(env, Z, policy_2, Gamma, Theta)
        printV(Z, env)

    printV(Z, env)
    ans=printPolicy(policy_2, env)

    printV(Z,env)
    table = tabulate(np.array(ans, dtype='object'), tablefmt="fancy_grid")

    # output
    print(table)
