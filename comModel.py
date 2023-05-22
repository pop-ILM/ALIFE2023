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
percent       = 1
spatial       = False
verbose       = False
xs = []
ys = []
stabilities = []
weights1 = []
weights2 = []
stabMatrix = [[0 for x in range(30)] for y in range(30)] 
firstSpoken = 0
speakingMatrix = []

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
    def __init__(self, id, model, signalSize, meaningSize, hiddenSize, utterances = 50,epochs = 100):
        super().__init__(id,model)
        self.id             = id
        self.expressiveness = 0
        self.age            = 0
        self.meaningSize    = meaningSize
        self.signalSize     = signalSize
        self.hiddenSize     = hiddenSize
        self.meaningSpace   = generate_all_signals(meaningSize)
        self.signalSpace    = generate_all_signals(signalSize)
        self.W1             = np.random.randn(signalSize[0], hiddenSize)
        self.W2             = np.random.randn(hiddenSize, meaningSize[0])
        self.lr             = 0.1
        self.utterances     = utterances
        self.epochs         = epochs
        self.stepVal        = 0
        self.meaningSignalPairings = {}
        self.spoke          = False
        self.recieved       = False



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
        global spatial
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        for agent in self.model.grid.get_cell_list_contents(neighbors_nodes):
            if agent.age<6 or adultComm:
                p = self.random.uniform(0, 1)
                if spatial:
                    distance = abs(math.floor((agent.id-1)/5) - math.floor((self.id-1)/5))
                    if distance == 4:
                        distance = 2
                    if distance == 5:
                        distance = 1
                    if((math.floor((self.id-1)/5) == math.floor((agent.id-1)/5)) or (p<=pow(probability,distance))):
                        agentComms.append(agent)
                else:
                    if((math.floor((self.id-1)/5) == math.floor((agent.id-1)/5)) or p<=probability):
                        agentComms.append(agent)
        #print(len(agentComms))
        if (len(agentComms)!=0):
            for i in range(int(self.epochs/len(agentComms))):
                np.random.shuffle(meanings)
                neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
                for agent in agentComms:
                    agent.induce(self.id,self.obvert(meanings))
            for agent in agentComms:
                oldComparison = compareStability(self,agent,self.meaningSpace)
                agent.induceUpdate(self.id,self.obvert(meanings))
                global verbose
                if verbose:
                    print(self.stepVal, ": Stability changed from ", oldComparison, " to ",compareStability(self,agent,self.meaningSpace))


    def induce(self, otherAgent,pairs):
        self.learnPairings(pairs)
        self.learnSignal(pairs)
        self.recieved = True

    def induceUpdate(self, otherAgent,pairs):
        if(self.recieved):
            oldExpressiveness = self.expressiveness
            self.updateMeaningSignalPairs()
            global verbose
            if verbose:
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
        stabilities[self.stepVal][self.id-1]=stability/total

    def updateMeaningSignalPairs(self):
        uniqueSignals = []
        meaningSignalPairings = {}

        sum= 0
        combined_matrix = normalize(self.confidence, axis=1, norm='l1')
        for m in range(len(self.meaningSpace)):
            sum+=combined_matrix[m][1]
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
        total  = 0
        for pair1 in self.meaningSignalPairings.keys():
            for pair2 in self.meaningSignalPairings.keys():
                if pair1 != pair2:
                    hdM = hammingDistance(pair1,pair2)
                    hdS = hammingDistance(self.meaningSignalPairings[pair1], self.meaningSignalPairings[pair2])
                    if(hdM == hdS):
                        output += 1
                    total += 1
                    print("Hamming Distance: Meaning: ",hdM," - Signal: ",hdS)

        return output/total

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
            if(len(possibleMatrix)>=6):
                speakingMatrix = random.sample(possibleMatrix, 6)	
                # print(self.stepVal,possibleMatrix,speakingMatrix)
        if speakingMatrix.count(self.id)>0:
            self.invent()
        self.age += 1
        if ((self.stepVal%6)*5+math.floor((self.stepVal%30)/6))%30 == self.id%30 and self.stepVal<960:
            self.witnesses = np.zeros((len(self.meaningSpace),len(self.signalSpace)))
            self.W1             = np.random.randn(self.signalSize[0], self.hiddenSize)
            self.W2             = np.random.randn(self.hiddenSize, self.meaningSize[0])
            self.age = 0
            # print(self.stepVal," : ","AGENT ",self.id," DIED")
        
        if self.age == 5:
            self.generateObvert()
        global expressivenesses
        expressivenesses[self.stepVal][self.id-1]=self.expressiveness
        self.gatherNearbyStability()
        self.stepVal+=1
        # if(self.stepVal%200==0):
        #     name = str(self.stepVal) +'.agentW1-'+str(self.id)+'.csv'
        #     np.savetxt(name,self.W1, delimiter=',')
        #     name = str(self.stepVal) +'.agentW2-'+str(self.id)+'.csv'
        #     np.savetxt(name,self.W2, delimiter=',')
        #     print(str(self.id) + ':' + str(self.expressiveness))
        global stabMatrix

        if(self.stepVal%1000==999):
            neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
            stabMatrix[self.id-1][self.id-1] = 1
            for agent in self.model.grid.get_cell_list_contents(neighbors_nodes):
                stabMatrix[self.id-1][agent.id-1]=1-compareStability(self,agent,self.meaningSpace)
        if(self.stepVal%1000==0):
            if self.id == 1:
                global probability
                global percent
                with open("data/stability"+str(percent)+".csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    flat = [item for sublist in stabMatrix for item in sublist]
                    print(flat)
                    writer.writerow(flat)







class LearningModel(Model):
    def __init__(self,N, signalSize, meaningSize, hiddenSize):
        self.N = N
        G = nx.complete_graph(30)
        self.G = G
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        for i, node in enumerate(self.G.nodes()):
            a = LearningAgent(i+1, self, signalSize, meaningSize, hiddenSize)
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
    global spatial
    name = str(N)+'-'+str(iterations)+('-s' if spatial else "")+'.csv'
    np.savetxt('data/expr-'+name,expressivenesses, delimiter=',')
    np.savetxt('data/stab-'+name,stabilities, delimiter=',')
    meanExpr = np.mean(expressivenesses,axis=1)
    meanStab = np.mean(stabilities,axis=1)
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
    print('python comModel.py -p percent | -s | -v | -h')
    print("")
    print("-p percent ~ The percent quantity of language being external. Default is 1.")
    print("-s ~ Spatial. Whether the community structure is spatial. Default is Off.")
    print("-v ~ Verbose. Adds more print tracking of agents communication, stability, expressiveness and timer. Default is Off.")
    print("-h ~ Help. Prints options.")
    print("")

def main(argv):
    short_options = "p:svh"
    try:
        arguments, values = getopt.getopt(argv, short_options)
    except getopt.error as err:
        # Output error, and return with an error code
        usage()
        print (str(err))
        sys.exit(2)
    global probability
    global percent
    global spatial
    for a, v in arguments:
        if a == ("-p"):
            percent = int(v)
        elif a == ("-s"):   
            spatial = True
        elif a == ("-v"):
            global verbose
            verbose = True
        elif a == ("-h"):
            usage()
            sys.exit(2)


    global adultComm
    
    global xs
    global ys
    global stabilities
    global weights1
    global weights2
    global stabMatrix
    global firstSpoken
    global speakingMatrix
    adultComm     = True
    xs = []
    ys = []
    stabilities = []
    weights1 = []
    weights2 = []
    stabMatrix = [[0 for x in range(30)] for y in range(30)] 
    firstSpoken = 0
    speakingMatrix = []
    x=percent*0.01
    if spatial:
        probability = -(4.1997 *(x - 1))/pow((-200 * x*x*x - 2100 * x*x + math.sqrt(pow((-200 * x*x*x - 2100 * x*x + 4800 * x - 2500),2) + 500000 * pow((x - 1),6)) + 4800 * x - 2500),(1/3)) + (0.052913 *pow((-200 *x*x*x - 2100 *x*x + math.sqrt(pow((-200 * x*x*x - 2100 *x*x + 4800 *x - 2500),2) + 500000 * pow((x - 1),6)) + 4800 * x - 2500),(1/3)))/(x - 1) - 0.66667
    else:
        probability = float((x*-4)/(25*x-25))
    print("perc:",percent," prob:",probability)
    _ = runSimulation(30,1001)
    inStab,outStab = calculate_stabilities()
    print(probability,":",inStab,"-",outStab)    



if __name__ == "__main__":
    main(sys.argv[1:])

