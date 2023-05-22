import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
import getopt, sys
from sklearn.preprocessing import normalize
import math
import random
import csv

probability   = 0.01
percent       = 0

death         = True
forced_obvert = False
bayes         = False
witnesses     = False
cont          = False
adultComm     = True
contVal       = 0
xs = []
ys = []
stabilities = []
weights1 = []
weights2 = []
stabMatrix = [[0 for x in range(30)] for y in range(30)] 
firstSpoken = 0
speakingMatrix = []
hammingVals = []

utt = 50

def compareStability(adultAgent,learnerAgent,meaningSpace):
    sum=0
    for m in meaningSpace:
        if(not np.array_equiv(learnerAgent.meaningSignalPairings[tuple(m)],(adultAgent.meaningSignalPairings[tuple(m)]))):
            sum+=1
    return(1-sum/len(meaningSpace))

# Sigmoid function
def sigmoid(a):
    return 1/(1 + np.exp(-a))

def mean_squared_error(preds,targets):
    return 0.5*(np.sum(preds-targets)**2)

def dLoss_dPred(preds,targets):
    return preds - targets

def dSigmoid_dA(a):
    return sigmoid(a)*(1-sigmoid(a))

def hammingDistance(a,b):
    output = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            output+=1
    return output

def generate_all_signals(size):
    quantity = np.power(size[1], size[0])
    output = np.zeros((quantity,size[0]))
    for i in range(quantity):
        total = i
        for j in range(size[0]-1,-1,-1):
            scale = np.power(size[1], (j))
            output[i][j] = int(total / scale) % size[1]
    return output

# Generates a random meaning from the meaning space
def generate_meaning(size,quantity):
    return np.random.randint(size[1],size = (quantity,size[0]))

class LearningAgent(Agent):
    def __init__(self, id, model, signalSize, meaningSize, hiddenSize, utterances = 50,epochs = 20):
        super().__init__(id,model)
        self.id             = id
        self.expressiveness = 0
        self.age            = 0
        self.meaningSize    = meaningSize
        self.signalSize     = signalSize
        self.hiddenSize     = hiddenSize
        self.meaningSpace   = generate_all_signals(meaningSize)
        self.signalSpace    = generate_all_signals(signalSize)
        self.W1             = np.random.randn(self.signalSize[0], self.hiddenSize)/self.signalSize[0]
        self.W2             = np.random.randn(self.hiddenSize, self.meaningSize[0])/self.hiddenSize
        self.lr             = 0.1
        self.utterances     = utterances
        self.witnesses      = np.zeros((len(self.meaningSpace),len(self.signalSpace)))
        self.epochs         = epochs
        self.stepVal        = 0
        self.meaningSignalPairings = {}
        self.spoke          = False
        self.recieved       = False
        if cont:
            self.stepVal = contVal + 1
            self.W1 = np.genfromtxt(str(contVal)+'.agentW1-' + str(self.id) + '.csv', delimiter=',')
            self.W2 = np.genfromtxt(str(contVal)+'.agentW2-' + str(self.id) + '.csv', delimiter=',')
            # self.W1 = np.genfromtxt('agentW1-' + str(self.id) + '.csv', delimiter=',')
            # self.W2 = np.genfromtxt('agentW2-' + str(self.id) + '.csv', delimiter=',')
            self.generateObvert()
            self.updateMeaningSignalPairs()
            self.age = 6
            #print(str(self.id) + ':' + str(self.expressiveness))


    def backpropagate(self, inputs, z, h,v, preds, targets):
        dS_dPreds    = dSigmoid_dA(v)
        dLoss_dP     = dLoss_dPred(preds, targets)
        dLoss_dV     = np.multiply(dS_dPreds, dLoss_dP)
        dLoss_dH     = np.matmul(dLoss_dV, self.W2.T)
        dLoss_dW2    = np.matmul(h.T, dLoss_dV)
        dS_dZ        = dSigmoid_dA(z)
        dLoss_dZ     = np.multiply(dS_dZ, dLoss_dH)
        dLoss_dW1    = np.matmul(np.asarray(inputs).T, dLoss_dZ)
        return dLoss_dW1, dLoss_dW2

    def learnPairings(self, pairings):
        signals = []
        meanings = []
        for i in pairings:
            signals.append(i[1])
            meanings.append(i[0])
        z,h,v,preds = self.forward(signals)
        dLoss_dW1,dLoss_dW2 = self.backpropagate(signals,z,h,v,preds,meanings)
        self.W1 -= self.lr*dLoss_dW1
        self.W2 -= self.lr*dLoss_dW2
        loss = mean_squared_error(preds,meanings)
        #print("loss", loss)
        return loss

    def invent(self):
        meanings = generate_meaning(self.meaningSize,self.utterances)
        global probability
        agentComms = []
        global adultComm
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        for agent in self.model.grid.get_cell_list_contents(neighbors_nodes):
            if agent.age<6 or adultComm:
                p = self.random.uniform(0, 1)
                agentComms.append(agent)
        #print(len(agentComms))
        if (len(agentComms)!=0):
            for i in range(int(self.epochs)):
                np.random.shuffle(meanings)
                neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
                for agent in agentComms:
                    agent.induce(self.id,self.obvert(meanings))
            for agent in agentComms:
                oldComparison = compareStability(self,agent,self.meaningSpace)
                agent.induceUpdate(self.id,self.obvert(meanings))
                #print(self.stepVal, ": Stability changed from ", oldComparison, " to ",compareStability(self,agent,self.meaningSpace))


    def induce(self, otherAgent,pairs):
        self.learnPairings(pairs)
        if(not forced_obvert):
            self.learnSignal(pairs)
        self.recieved = True

    def induceUpdate(self, otherAgent,pairs):
        if(self.recieved):
            oldExpressiveness = self.expressiveness
            self.updateMeaningSignalPairs()
            print(self.stepVal, ": ", otherAgent," spoke to ", self.id, " expressiveness updated from ",oldExpressiveness," to ", self.expressiveness)
        self.recieved = False

    def obvert(self,suppliedMeanings):
        pairings = []
        for i in suppliedMeanings:
            pairings.append(tuple((i,self.meaningSignalPairings[tuple(i)])))
        return pairings

    def learnSignal(self,pairs):
        #self.confidence *= 0.9
        signals = []
        meanings = []
        if witnesses:
            self.witnesses*=0.99
        for i in pairs:
            meanings.append(i[0])
            signals.append(i[1])
        _,_,_,preds = self.forward(signals)
        for p in range(len(signals)):
            mi=0
            si=0
            for m in range(len(self.meaningSpace)):
                if((self.meaningSpace[m] == meanings[p]).all()):
                    mi=m
            for s in range(len(self.signalSpace)):
                if((self.signalSpace[s] == signals[p]).all()):
                    si=s

            if witnesses:
                self.witnesses[mi][si]+=1
            else:
                #self.confidence[:][si]  *=0.99
                #self.confidence[mi]     *=0.99
                self.confidence[mi][si] =1
                for i in range(len(self.meaningSpace[0])):
                    if(self.meaningSpace[mi][i] == 0):
                        self.confidence[mi][si]*=(1-preds[p][i])
                    else:
                        self.confidence[mi][si]*=(preds[p][i])

    def generateObvert(self):
        _,_,_,PMeanings = self.forward(self.signalSpace)
        confidence = np.ones((len(self.meaningSpace),len(self.signalSpace)))
        pairings = []
        for m in range(len(self.meaningSpace)):
            for s in range(len(self.signalSpace)):
                for i in range(len(self.meaningSpace[0])):
                    if(self.meaningSpace[m][i] == 0):
                        confidence[m][s]*=(1-PMeanings[s][i])
                    else:
                        confidence[m][s]*=(PMeanings[s][i])
            #print(str(meaningSpace[m]) + " - " + str(meaningSignalPairings[tuple(meaningSpace[m])]))
        self.confidence = confidence

    def gatherNearbyStability(self):
        stability = 0
        total = 0
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        for agent in self.model.grid.get_cell_list_contents(neighbors_nodes):
            stability += compareStability(self,agent,self.meaningSpace)
            total+=1
        stabilities[self.stepVal][self.id-1]=1-(stability/total)

    def updateMeaningSignalPairs(self):
        uniqueSignals = []
        meaningSignalPairings = {}

        sum= 0
        if forced_obvert:
            combined_matrix = self.confidence
        else:
            if witnesses:
                normalised_w = normalize(self.witnesses, axis=1, norm='l1')
                combined_matrix = normalize((self.confidence + normalised_w), axis=1, norm='l1')
            else:
                combined_matrix = normalize(self.confidence, axis=1, norm='l1')
            if bayes:
                Ssum = np.sum(combined_matrix,axis=0)
                Msum = np.sum(combined_matrix,axis=1)
                S = Ssum/np.sum(Ssum)
                M = Msum/np.sum(Msum)
                print(M[1],S[1])
        for m in range(len(self.meaningSpace)):
            sum+=combined_matrix[m][1]
            if bayes:
                signal = self.signalSpace[np.argmax(combined_matrix[m])]
            else:
                signal = self.signalSpace[np.argmax(combined_matrix[m])]

            meaningSignalPairings[tuple(self.meaningSpace[m])] = signal
            if tuple(signal) not in uniqueSignals:
                uniqueSignals.append(tuple(signal))

        self.uniqueSignals         = uniqueSignals
        self.meaningSignalPairings = meaningSignalPairings
        self.expressiveness        = len(uniqueSignals)/len(self.signalSpace)




    def forward(self,u):
        z = np.matmul(u,self.W1)
        h = sigmoid(z)
        v = np.matmul(h,self.W2)
        output = sigmoid(v)
        return z,h,v,output

    def analyseHammingData(self):
        output = 0
        total  = np.zeros((8,8))
        totalNegative = np.zeros((8,8))
        for key in self.meaningSignalPairings.keys():
            val = self.meaningSignalPairings[key]
            for i in range(len(key)):
                for j in range(len(val)):
                    if(key[i]==val[j]):
                        total[i][j]+=1
                    else:
                        totalNegative[i][j]+=1

        total = total/256
        totalNegative = totalNegative/256
        for i in range(8):
            output+=max(max(total[i]),max(totalNegative[i]))
        print(output/8)
        return output/8

    def printLanguage(self):
        for i in self.meaningSignalPairings:
            print(i," - ",self.meaningSignalPairings[i])

    def step(self):
        p = self.random.uniform(0, 1)
        global firstSpoken
        global speakingMatrix
        if(self.stepVal>firstSpoken):
            #if(self.stepVal%100 ==0):
                #print(self.stepVal)
            speakingMatrix = []
            possibleMatrix = []
            firstSpoken = self.stepVal
            neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
            for agent in self.model.grid.get_cell_list_contents(neighbors_nodes):
                if agent.age>5:
                    possibleMatrix.append(agent.id)
            if(len(possibleMatrix)>=1):
                speakingMatrix = random.sample(possibleMatrix, 1)	
                #print(self.stepVal,possibleMatrix,speakingMatrix)
        if speakingMatrix.count(self.id)>0:
            self.invent()
        self.age += 1
        if ((self.stepVal%10) == self.id%10) and self.stepVal<90:
            self.witnesses = np.zeros((len(self.meaningSpace),len(self.signalSpace)))
            self.W1             = np.random.randn(self.signalSize[0], self.hiddenSize)/self.signalSize[0]
            self.W2             = np.random.randn(self.hiddenSize, self.meaningSize[0])/self.hiddenSize
            self.age = 0
            print(self.stepVal," : ","AGENT ",self.id," DIED")
        
        if self.age == 5:
            self.generateObvert()
        global expressivenesses
        expressivenesses[self.stepVal][self.id-1]=self.expressiveness
        self.gatherNearbyStability()
        self.stepVal+=1
        if forced_obvert and self.spoke == True:
            self.W1             = np.random.randn(self.signalSize[0], self.hiddenSize)
            self.W2             = np.random.randn(self.hiddenSize, self.meaningSize[0])
            self.spoke = False
        if(self.stepVal==99):
            global hammingVals
            hammingVals.append(self.analyseHammingData())
        # if(self.stepVal%200==0):
        #     name = str(self.stepVal) +'.agentW1-'+str(self.id)+'.csv'
        #     np.savetxt(name,self.W1, delimiter=',')
        #     name = str(self.stepVal) +'.agentW2-'+str(self.id)+'.csv'
        #     np.savetxt(name,self.W2, delimiter=',')
        #     print(str(self.id) + ':' + str(self.expressiveness))






class LearningModel(Model):
    def __init__(self,N, signalSize, meaningSize, hiddenSize, avg_node_degree=3):
        self.N = N
        prob = avg_node_degree / N
        #G = nx.erdos_renyi_graph(n=self.N, p=prob)
        G = nx.complete_graph(10)
        # while not nx.is_connected(G):
        #     G = nx.erdos_renyi_graph(n=self.N, p=prob)
        self.G = G
        self.grid = NetworkGrid(self.G)

        global utt
        self.schedule = RandomActivation(self)
        for i, node in enumerate(self.G.nodes()):
            a = LearningAgent(i+1, self, signalSize, meaningSize, hiddenSize,utterances=utt)
            a.generateObvert()
            a.updateMeaningSignalPairs()
            self.schedule.add(a)
            self.grid.place_agent(a, node)
        # nx.draw(self.G, with_labels=True, font_color="whitesmoke")
        # plt.show()

        self.datacollector = DataCollector(
            agent_reporters={"expressiveness": "expressiveness"})


    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.update()

    def update(self):
        pass



def runSimulation(N,iterations):
    model = LearningModel(N,(8,2),(8,2),8)
    i=0
    global xs
    global expressivenesses
    global stabilities
    global weights1
    global weights2
    xs = np.arange(iterations)
    expressivenesses = np.zeros((iterations,N))
    stabilities = np.zeros((iterations,N))


    while i<iterations:
        i += 1
        model.step()
    modelDF = model.datacollector.get_agent_vars_dataframe()
    # plt.plot(xs,ys,linewidth=0.5)
    # plt.title('Average expressiveness of ' + str(N) + ' simulated agents')
    # plt.ylim((0,1))
    # plt.ylabel('Average expressiveness')
    # plt.xlabel('Iterations')
    # plt.show()

    # name = str(N)+'-'+str(iterations)+('-d' if death else '')+('-o' if forced_obvert else '')+('-w' if witnesses else '')+'.csv'
    global hammingVals
    print("conf " + str(np.average(hammingVals)))
    name = str(N)+'-'+str(utt)+'.csv'
    np.savetxt('data/expr-'+name,expressivenesses, delimiter=',')
    np.savetxt('data/stab-'+name,stabilities, delimiter=',')
    # with open("stab.csv", "a", newline="") as f:
    #                 writer = csv.writer(f)
    #                 writer.writerow(stabilities)
    # with open("expre.csv", "a", newline="") as f:
    #                 writer = csv.writer(f)
    #                 writer.writerow(expressivenesses)
   

    # for i in range(iterations):
    #     #print(i)
    # for i in range(iterations):
    #     #print(meanExpr[i])
    # for i in range(iterations):
        #print(meanStab[i])


    return modelDF.expressiveness
    
def calculate_stabilities():
    local_sum=0
    local_tot=0
    out_grp_sum=0
    out_grp_tot=0
    global stabMatrix
    for i in range(30):
        for j in range(30):
            if i != j:
            #print(agents[i].meaningSignalPairings)
                if(math.floor((i+1)/5)==math.floor((j+1)/5)):
                    local_sum+=stabMatrix[i][j]
                    local_tot+=1
                else:
                    out_grp_sum+=stabMatrix[i][j]
                    out_grp_tot+=1

    return local_sum/local_tot,out_grp_sum/out_grp_tot


def usage():
    print('python popModel.py -u utterances | -h')
    print("")
    print("-u utterances ~ The number of utterances spoken per generation. Default is 50.")
    print("-h ~ Help. Prints options.")
    print("")

def main(argv):        
    short_options = "u:h"
    try:
        arguments, values = getopt.getopt(argv, short_options)
    except getopt.error as err:
        # Output error, and return with an error code
        usage()
        print (str(err))
        sys.exit(2)

    for a, v in arguments:
        if a == ("-u"):
            global utt
            utt = int(v)
        elif a == ("-h"):
            usage()
            sys.exit(2)
    expressiveness = runSimulation(10,100)



if __name__ == "__main__":
    main(sys.argv[1:])

