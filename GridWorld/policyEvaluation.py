import numpy as np
import matplotlib.pyplot as plt
from Gridworld import printV , Gridworld

def evaluatePolicy(Grid, V, Policy, Gamma, Theta):
    converged=False
    iteration=0
    while not converged:
        Delta = 0
        for state in Grid.statespace:
            iteration += 1
            oldV = V[state]
            total = 0 
            #action_prob = 1/len(Policy[state])
            action_prob = 1
            for action in Policy[state]:
                for i in Grid.p[(state,action)]:
                    (tp,newState,reward,oldState,act)=i
                    #if oldState == state and act == action:
                    total += action_prob *  (reward +  tp * Gamma * oldV[newState])
            
            Delta = max(Delta,np.abs(oldV-V[state]))
            
            V[state] = total
            #printV(V,Grid)
            
            converged = True if Delta < Theta else False
    print(iteration,'sweeps of state space in policy evaluation')
    return V


# if __name__ == '__main__':
#     # map magic squares to their connecting square
#     #wall=[(12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13), (12, 14), (7, 2), (8, 2), (9, 2), (10, 2), (11, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10)]
#     wall=[(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10)]
#     env = Gridworld(9, 9, wall,0.2)
#     # print(env.getAgentRowAndColoumn())
#     # env.setState(1)
#     # print(env.getAgentRowAndColoumn())
#     env.renderHeatMap()
#     Gamma=1
#     Theta = 1e-6
#     V = {}
#     for state in env.statespaceplus:
#         V[state] = 0

#     policy = {}
#     for state in env.statespace:
#         policy[state] = env.possibleaction
#     New_V = evaluatePolicy(env, V, policy, Gamma, Theta)
