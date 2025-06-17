import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import pandas as pd
import statistics

def random_weight():
    if np.random.binomial(1,.2) == 1:
        return -0.2
    return 1

def sobreposition_matrix(assembly_list):
    df = pd.DataFrame()
    for i in range(0,len(assembly_list)):
        list_sobreposition = []
        for j in assembly_list:
            list_sobreposition.append(len([k for k in assembly_list[i] if k in j]))
        df[str(i)] = list_sobreposition
    #print(df)
    return df

class artificial_neuron:
    
    def __init__(self,index):

        self.f = False
        self.f_1 = False
        self.index = index
        self.synaptic_input = 0
        self.pre_synaptic_list = [] #dict {neuron : weight}
        
        ####################
        self.flag = False

    # INFO:
    def __str__(self):
        self.__show_att__()
        return ""

    def __repr__(self):
        return "Neuron {0}".format(self.index)
    
    def __show_att__(self):
        for i,j in self.__dict__.items():
            print("{0}:".format(i),j)
 
    # OPERATIONS:
    def __eq__(self, other_neuron):
        return self.index == other_neuron.index
    
    def __gt__(self, other_neuron):
        return self.synaptic_input > other_neuron.synaptic_input
    
    def __add__(pre_neuron, pos_neuron):
        pos_neuron.pre_synaptic_list.append({"neuron": pre_neuron, "w" : random_weight() })

    # INTEGRATION AND LEARNING:
    def synaptic_integration(self):
        self.synaptic_input = 0
        for i in self.pre_synaptic_list:
            self.synaptic_input += i["neuron"].f_1 * i["w"]
        #self.synaptic_input = np.array([self.pre_synaptic_list[i]['neuron'].f_1 * self.pre_synaptic_list[i]['w'] for i in range(len(self.pre_synaptic_list))]).sum()
    
    def update_weights(self, beta):
        for i in self.pre_synaptic_list:
            if i['neuron'].f_1 == True:
                i['w'] *= (1 + beta)

class artificial_area:

    def __init__(self,n,p,k,beta,d,tau,e_max, index):
        
        self.n = n
        self.p = p
        self.k = k
        self.beta = beta
        self.index = index
        self.e_max = e_max
        self.e_max_constant = (1 - d/tau)
        self.neuron_list = [artificial_neuron(i + index*n) for i in range(0,n)]
        #===========#
        self.neuron_winners = None
        self.neuron_winners_1 = None

        self.recovery_list = []
        self.recovery_fire = []
    # INFO:
    def __str__(self):
        self.__show_att__()
        return ""

    def __repr__(self):
        return "Area {0}".format(self.index)
    
    def __show_att__(self):
        for i,j in self.__dict__.items():
            print("{0}:".format(i),j)

    # OPERATIONS:
    def __add__(pre_area,pos_area):
        
        #pre_area = pos_area  --> recurrence
        #pre_area != pos_area --> afference
        
        for i in pre_area.neuron_list:
            for j in pos_area.neuron_list:
                if (np.random.binomial(1,pre_area.p) == 1 and i != j):
                    i + j

    # METHODS:
    def sample_random_set(self, size):
        return rnd.sample(self.neuron_list,size)

    def winners_list(self):
        for i in self.neuron_list:
            i.synaptic_integration()
            
        if self.e_max == False:
            self.neuron_list.sort(reverse=True)
            return [self.neuron_list[i] for i in range(0,self.k)]

        elif self.e_max == True:
            inputs_area = [i.synaptic_input for i in self.neuron_list]

            return [i for i in self.neuron_list if i.synaptic_input >= self.e_max_constant * max(inputs_area)]

def G_density(graph):
    vertex = len(graph)
    if vertex > 1:
        edge = 0
        for i in graph:
            for j in i.pre_synaptic_list:
                if j["neuron"] in graph:
                    edge += 1
        density = (edge)/(vertex*(vertex - 1))
        return density
    return 0

def FIRE(stimuli,B,T):
    for i in stimuli:
        i.f_1 = True

    for t in range(1,T+1):
            
        if t > 1:
            if B.neuron_winners != None:
                for i in B.neuron_winners:
                    i.f_1 = True
                B.neuron_winners_1 = B.neuron_winners

        new = 0
        B.neuron_winners = B.winners_list()
        for i in B.neuron_winners:
            i.update_weights(B.beta)
            if i.flag == False:     # D.
                i.flag = True       # D.
                new += 1            # D.

        if t == 1:
            new_1 = len(B.neuron_winners)
        else:
            new_1 = len([i for i in B.neuron_winners if i not in B.neuron_winners_1])

        print(t,new,new_1, len(B.neuron_winners))      # D.

        if t > 1:
            for i in B.neuron_winners_1:
                i.f_1 = False


    for i in stimuli:
        i.f_1 = False

    for i in B.neuron_list:
        i.flag = False ; i.f_1 = False

    assembly = B.neuron_winners
    B.neuron_winners = None ; B.neuron_winners_1 = None

    return assembly 

def FIRE_F(stimuli,B):
    t = 1
    for i in stimuli:
        i.f_1 = True

    while True:

        if t > 1:
            if B.neuron_winners != None:
                for i in B.neuron_winners:
                    i.f_1 = True
                B.neuron_winners_1 = B.neuron_winners

        new = 0
        B.neuron_winners = B.winners_list()
        for i in B.neuron_winners:
            i.update_weights(B.beta)
            if i.flag == False:
                i.flag = True
                new += 1

        if t == 1:
            new_1 = len(B.neuron_winners)
        else:
            new_1 = len([i for i in B.neuron_winners if i not in B.neuron_winners_1])
        print(t, new, new_1, len(B.neuron_winners))

        if t > 1:
            for i in B.neuron_winners_1:
                i.f_1 = False
        
            if B.e_max == True:
                if (new_1 == 0) and (len(B.neuron_winners) == len(B.neuron_winners_1)):
                    break
            else:
                if (new == 0):
                    break
            
        t +=1

    for i in stimuli:
        i.f_1 = False

    for i in B.neuron_list:
        i.flag = False ; i.f_1 = False

    assembly = B.neuron_winners
    B.neuron_winners = None ; B.neuron_winners_1 = None
    if G_density(assembly) > B.p and len(assembly) > 5:
        return assembly,t
    return None,None

def FIRE_R(stimuli, B, assembly, T):
    for i in stimuli:
        i.f_1 = True

    B.beta = 0

    for t in range(1,T+1):
            
        if t > 1:
            if B.neuron_winners != None:
                for i in B.neuron_winners:
                    i.f_1 = True
                B.neuron_winners_1 = B.neuron_winners

        B.neuron_winners = B.winners_list()
        for i in B.neuron_winners:
            i.update_weights(B.beta)

        print(t, len([i for i in B.neuron_winners if i in assembly]), len(B.neuron_winners))
        B.recovery_list.append(len([i for i in B.neuron_winners if i in assembly]))
        B.recovery_fire.append(len(B.neuron_winners))
        if t > 1:
            for i in B.neuron_winners_1:
                i.f_1 = False

    for i in stimuli:
        i.f_1 = False

    for i in B.neuron_list:
        i.flag = False ; i.f_1 = False

e_max = True
#n = 1000 ; p = .1 ; k = 37 ; beta = 0.05 ; T = 15 ; d = 3 ; tau = 30
n = 1000 ; p = .5 ; k = 200 ; beta = 0.01 ; T = 20 ; d = 3 ; tau = 30 
#inhibition == -.2
#beta = [.1,.05,.01,.005,.001]

# Multiple assemblies (same area)
'''
L = [17, 32, 35]
for j in L:
    S = artificial_area(n,p,k,beta,d,tau,e_max, 0)
    A = artificial_area(n,p,k,beta,d,tau,e_max, 1)

    S + A
    A + A
    assemblies = []
    stimuli = []
    for i in range(0,10):
        #print(i)
        set_test = S.sample_random_set(k)
        assembly,time_f = FIRE_F(set_test,A)
        if assembly != None:
            stimuli.append(set_test)
            assemblies.append(assembly)
            #print("Density = {0}".format(G_density(assembly)))
        #print()

    df_A = sobreposition_matrix(assemblies)
    df_S = sobreposition_matrix(stimuli)
    df_A.to_csv("matrix_assembly_emax_{0}".format(j))
    df_S.to_csv("matrix stimuli_emax_{0}".format(j))
    print(j)
'''

#Formation & convergence
'''
size_list = []
density_list = []
time_list = []
df_formation = pd.DataFrame()
df_recovery = pd.DataFrame()
print("BETA: ",beta)
for i in range(0,500):

    S = artificial_area(n,p,k,beta,d,tau,e_max, 0)
    A = artificial_area(n,p,k,beta,d,tau,e_max, 1)

    S + A
    A + A

    set_test = S.sample_random_set(k)
    assembly,time_f = FIRE_F(set_test,A)

    size_list.append(len(assembly))
    density_list.append(G_density(assembly))
    time_list.append(time_f)
    FIRE_R(set_test,A,assembly,T)
    df_recovery["recovery"] = A.recovery_list
    df_recovery["fire"] = A.recovery_fire
    df_recovery.to_csv("recovery_0_05_k_winners_" + str(i+1))
    print(i)


df_formation["size"] = size_list
df_formation["time"] = time_list
df_formation["density"] = density_list
df_formation.to_csv("formation_0_05_k_winners_")
'''

#RECOVERY:
'''
e_max = True                    
#n = 1000 ; p = .1 ; k = 37 ; beta = 0.01; T = 15 ; d = 3 ; tau = 30 # kwin
n = 1000 ; p = .5 ; k = 200 ; beta = 0.01; T = 15 ; d = 3 ; tau = 30 # emax
beta_list = [0.1,0.05,0.01,0.005,0.001]
beta_str = ["0_1","0_05","0_01","0_005","0_001"]
for j in range(0,len(beta_list)):
    print("Beta recovery: {0}".format(beta_list[j]))
    list_recovery = []
    i = 0
    while len(list_recovery)< 200:
        print("===ITERACAO {0}===".format(i))

        S = artificial_area(n,p,k,beta_list[j],d,tau,e_max, 0)
        A = artificial_area(n,p,k,beta_list[j],d,tau,e_max, 1)

        S + A
        A + A

        set_test = S.sample_random_set(k)
        assembly,time_f = FIRE_F(set_test,A)
        if assembly != None:
            FIRE_R(set_test,A,assembly,T)
            list_recovery.append(A.recovery_list[-1]/len(assembly))
            i += 1
    df_recovery = pd.DataFrame(list_recovery)
    df_recovery.to_csv("recovery_e_max_" + str(beta_str[j]) + "_" + str(T))
'''

#Size stimuli & Prob:
'''
n = 1000 ; p = .5 ; k = 200 ; beta = 0.01 ; T = 15 ; d = 3 ; tau = 30
e_max = True


size_k_s = [50,100,150,250,500]
for size in size_k_s:
    size_list = []
    print("Size stimuli: ",size)
    for i in range(0,200):
        print(i)

        S = artificial_area(n,p,size,beta,d,tau,e_max, 0)
        A = artificial_area(n,p,size,beta,d,tau,e_max, 1)

        S + A
        A + A

        set_test = S.sample_random_set(size)
        assembly,time_f = FIRE_F(set_test,A)
        if assembly != None:
            size_list.append(len(assembly))
    df_size_list = pd.DataFrame(size_list)
    df_size_list.to_csv("size_stimuli_" + str(size))
'''
    