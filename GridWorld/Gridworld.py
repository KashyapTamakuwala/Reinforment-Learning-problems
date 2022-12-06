from os import stat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.palettes import set_color_codes


#add wall in state space

class Gridworld(object):
    def __init__(self,m,n,wall,alpha):
        self.grid = np.zeros((m,n))
        self.n = n # no of coloumn in grid world
        self.m = m # no of rows in grid world
        self.statespace = [i for i in range(self.m*self.n)]
        self.statespace.remove(self.m*self.n - 1) # all state space except terminal
        self.statespaceplus = [i for i in range(self.m*self.n)] # all state space including terminal
        self.actionspace = {'↑': -self.m, '↓':self.m,
                            '→':1, '←':-1}
        self.possibleaction = ['↑','↓','←','→']
        self.addWalls(wall)
        self.agentposition = 0
        self.p={}
        self.alpha = alpha
        self.initP()
        
    
    def addWalls(self, wall):
        self.wall=[((self.m*i[0]) + i[1]) for i in wall]
        i=2
        for w in self.wall:
            x= w // self.m
            y= w % self.n
            self.grid[x,y] = i

    def initP(self):
        for state in self.statespace:
            for action in self.possibleaction:
                
                reward = -1
                tp = 1-self.alpha
                newState= state + self.actionspace[action]
                
                if newState in self.wall:
                    reward = -1000
                    tp=1
                    newState=state
                if self.offGridMove(newState,state):
                    reward=-1000
                    newState=state
                if self.isTerminalState(newState):
                    tp=1
                    reward = 0
                
                self.p[(state,action)] = [(tp,newState,reward,state,action)] # correct new state  after taking action on oldstate and its transition probability
                
                for a in self.possibleaction:
                    reward = -1
                    if a != action:
                        tp = self.alpha / 4
                        wrongState = state + self.actionspace[a]
                        if wrongState in self.wall:
                            reward = -1000
                        if self.offGridMove(wrongState,state):
                            reward=-1000
                            wrongState=state
                        if self.isTerminalState(wrongState):
                            reward = 0
                            tp=1
                        self.p[(state,action)].append((tp,wrongState,reward,state,action)) # wrong new state  after taking action on oldstate and its transition probability



    
    def isTerminalState(self , state):
        return state in self.statespaceplus and state not in self.statespace

    def getAgentRowAndColoumn(self):
        x = self.agentposition // self.m
        y = self.agentposition % self.n
        
        return x,y

    def setState(self,state):
        x, y = self. getAgentRowAndColoumn()
        self.grid[x][y] = 0
        self.agentposition = state
        x, y = self.getAgentRowAndColoumn()
        self.grid[x][y] = 1

    def offGridMove(self,newState,oldState):
        if newState not in self.statespace:
            return True
        elif oldState % self.m == 0 and newState % self.m == self.m -1:
            return True
        elif oldState % self.m == self.m - 1 and newState % self.m ==0:
            return True
        return False

    def step(self,action):
        w=False
        x, y = self.getAgentRowAndColoumn()
        resultingState = self.agentposition + self.actionspace[action]
        if resultingState in self.wall:
            w=True
            resultingState=self.agentposition
        
        if w:
            reward = -1000
        elif self.isTerminalState(resultingState):
            reward = 0
        else:
            reward = -1
        
        if not self.offGridMove(resultingState,self.agentposition):
            self.setState(resultingState)
            return resultingState,reward,self.isTerminalState(self.agentposition),None
        else:
            return self.agentposition,reward,self.isTerminalState(self.agentposition),None
        
    def reset(self):
        self.agentposition = 0 
        self.grid = np.zeros(self.m,self.n)
        self.addWalls(self.wall)
        return self.agentposition
    
    def render(self):
        print("-"*100)
        print(type(self.grid))
        for row in self.grid:
            for col in row:
                if col == 0:
                    print('-',end='\t')
                elif col == 1:
                    print('X',end='\t')
                elif col == 2:
                    print('W',end='\t')
            print('\n')
        print('-'*100)

    def renderHeatMap(self):
        # plt.imshow(self.grid, cmap='hot')
        sns.heatmap(self.grid, linewidth=0.5)
        plt.show()


def printV(V , grid):
    disp=np.zeros((grid.m,grid.n))
    print("-"*100)
    for idx, row in enumerate(grid.grid):
        for idy, _ in enumerate(row):
            state=grid.m * idx + idy
            #print('%.2f' % V[state], end='\t')
            disp[idx][idy]=V[state]
        #print('\n')
    # sns.heatmap(disp, linewidth=0.5)
    # plt.show()
    print("-"*100)
    return disp
    
def printPolicy(policy , grid):
    print("-"*100)
    lis=[]
    for idx, row in enumerate(grid.grid):
        temp=[]
        for idy, _ in enumerate(row):
            state=grid.m * idx + idy
            if not grid.isTerminalState(state):
                if state not in grid.wall:
                    temp.append(policy[state])
                    #print('%s' % policy[state], end='\t')
                else:
                    temp.append("--")
                    #print('%s' % '--', end='\t')
            else:
                temp.append("--")
                #print('%s' % '--', end='\t')   
        #print('\n')
        lis.append(temp)
    print("-"*100)
    return lis


# if __name__ == '__main__':
#     # map magic squares to their connecting square
#     wall=[(12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13), (12, 14), (7, 2), (8, 2), (9, 2), (10, 2), (11, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10)]
#     env = Gridworld(15, 15, wall,0.2)
#     print(env.getAgentRowAndColoumn())
#     env.setState(1)
#     print(env.getAgentRowAndColoumn())
#     env.renderHeatMap()
