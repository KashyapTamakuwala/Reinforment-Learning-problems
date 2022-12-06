import numpy as np
import matplotlib.pyplot as plt
from Gridworld import  Gridworld , printPolicy , printV
from tabulate import tabulate
from policyEvaluation import evaluatePolicy

def improvePolicy(Grid,V,Policy,Gamma):
    
    stable = True
    newPolicy = {}
    iteration = 0
    
    for state in Grid.statespace:
        iteration += 1
        oldAction = Policy[state]
        value = []
        newAction = []
    
        for action in Policy[state]:
            action_prob = 1
            temp=0
    
            for i in Grid.p[(state,action)]:
                (tp,newState,reward,oldState,act)=i
                temp = temp + action_prob * tp  * (reward +  Gamma *   V[newState])
                break
            value.append(temp)
            newAction.append(action)
        
        value = np.array(value)
        best = np.where(value == value.max())[0]
        #print(best)
        bestAction = [newAction[item] for item in best]
        newPolicy[state] = bestAction

        if oldAction != bestAction:
            stable = False
    
    print(iteration,'sweeps of state space in policy improvement')
    return stable, newPolicy


if __name__ == '__main__':
    # map magic squares to their connecting square
    wall=[(12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13), (12, 14), (7, 2), (8, 2), (9, 2), (10, 2), (11, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10)]
    #wall=[(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10)]
    env = Gridworld(15, 15, wall,0.2)
    # print(env.getAgentRowAndColoumn())
    # env.setState(1)
    # print(env.getAgentRowAndColoumn())
    env.renderHeatMap()
    Gamma=1
    Theta = 1e-6
    stable = False

    V = {}
    for state in env.statespaceplus:
        V[state] = 0

    policy = {}
    for state in env.statespace:
        policy[state] = env.possibleaction

    k=0
    while not stable:
        print(k)
        k=k+1
        New_V = evaluatePolicy(env, V, policy, Gamma, Theta)
        printV(New_V, env)
        stable,policy = improvePolicy(env,V,policy,Gamma)
        print(stable)
        arr=printPolicy(policy,env)

    printV(New_V,env)
    table = tabulate(np.array(arr, dtype='object'), tablefmt="fancy_grid")

    # output
    print(table)
