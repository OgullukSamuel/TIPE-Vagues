import tensorflow as tf
import numpy as np
import os
import h5py
import utilities
import parametres
import environnement
from itertools import count

class Agent(tf.Module):  
    """
    création d'un réseau de neurone profond 
    """
    def __init__(self,env):
        param=parametres.para()
        self.env=env
        self.tau_update=param.tau_update
        self.dueling=param.dueling
        self.TAU = param.tau
        self.inputnb=param.inputnb
        self.actionspace=param.actionspace
        self.initializer=param.initializer
        self.supervised=param.supervised
        self.named=param.named
        self.opti=param.opti
        self.lossing=param.lossing
        self.dir=param.dir
        self.metrique=param.metrique
        self.tau_update=param.tau_update
        self.gamma=param.gamma
        self.episodes =param.episodes

        self.batch_size=param.batch_size
        
        self.memory_size=param.memory_size
        self.a=param.PER_alpha
        self.e=param.PER_epsi
        self.tree = utilities.SumTree(self.memory_size)

        self.start = param.eps_st										
        self.end = param.eps_end								
        self.decay = param.eps_decay

        self.current_step = 0
        self.num_actions = param.action_space

        self.summon_net("Policy")
        self.summon_net("Target")

        if not os.path.exists(self.dir): os.makedirs(self.dir)
        self.saveplace1 = os.path.join(self.dir,"Policy_network"+".h5") 
        self.saveplace2 = os.path.join(self.dir,"Target_network"+".h5") 

    def summon_net(self,name):
        if not self.dueling:
            model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=3,kernel_size=2, strides=1, activation='relu',input_shape=(self.inputnb[0],self.inputnb[1],self.inputnb[2])),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding='valid'),
            tf.keras.layers.PRelu(),
            tf.keras.layers.Conv2D(filters=2,kernel_size=2, strides=2, activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024,activation="elu", kernel_initializer=self.initializer),
            tf.keras.layers.Dense(512, activation="elu", kernel_initializer=self.initializer),
            tf.keras.layers.Dense(256, activation='elu', kernel_initializer=self.initializer),
            tf.keras.layers.Dense(128, activation='elu', kernel_initializer=self.initializer),
            tf.keras.layers.Dense(64,  activation='elu', kernel_initializer=self.initializer),
            tf.keras.layers.Dense(32,  activation='elu', kernel_initializer=self.initializer),
            tf.keras.layers.Dense(self.actionspace, activation='linear', kernel_initializer=self.initializer)])

            model.compile(optimizer=self.opti, loss=self.lossing, metrics=self.metrique)
            model.summary()
        else:
            input=tf.keras.layers.Input(shape=self.inputnb,name="Input")

            convo = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=3,kernel_size=2, strides=1, activation="relu",input_shape=(self.inputnb[0],self.inputnb[1],self.inputnb[2])),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding="valid"),
                tf.keras.layers.Conv2D(filters=2,kernel_size=2, strides=2, activation="relu"),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"),
                tf.keras.layers.PReLU(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1024,activation="relu", kernel_initializer=self.initializer)],name="Convolution")(input)

            value_net = tf.keras.Sequential([
                tf.keras.layers.Dense(128,activation="relu", kernel_initializer=self.initializer),
                tf.keras.layers.Dense(1, kernel_initializer=self.initializer),
                tf.keras.layers.Lambda(lambda s: tf.keras.backend.expand_dims(s[:, 0], -1),output_shape=(self.actionspace,))],name="Value_network")(convo)

            advantage_net = tf.keras.Sequential([ 
                tf.keras.layers.Dense(128, activation="relu", kernel_initializer=self.initializer),
                tf.keras.layers.Dense(self.actionspace, kernel_initializer=self.initializer),
                tf.keras.layers.Lambda(lambda a: a[:, :] - tf.keras.backend.mean(a[:, :], keepdims=True),output_shape=(self.actionspace,))],name="Advantage_network")(convo)

            add = tf.keras.layers.Add()([value_net, advantage_net])
            output = tf.keras.layers.Dense(self.actionspace, kernel_initializer=self.initializer)(add)

            model = tf.keras.Model(input, output,name=self.named)
            model.compile(optimizer=self.opti, loss=self.lossing, metrics=self.metrique)
            tf.keras.utils.plot_model(model, to_file='réseau.png')
            model.summary()
        if name=="Target": self.targetnet=model
        else:self.policynet=model

    def tautransfer(self):
        if not self.tau_update:
            self.transfer(self.policy_model)
        else:
            q_model_theta = self.policynet.get_weights()
            target_model_theta = self.targetnet.get_weights()
            i = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                target_model_theta[i] = target_weight
                i += 1
            self.targetnet.set_weights(target_model_theta)

    def transfer(self,model):
        self.targetnet.set_weights(self.policynet.get_weights())

    def fitting(self):
        experiences = self.samplememory(self.batchsize)
        states, actions, rewards , next_states, dones = utilities.extract_tensors(list(zip(*experiences))[1])

        target = tf.squeeze(self.policynet.predict(states)).numpy()
        target_next = tf.squeeze(self.policynet.predict(next_states)).numpy()
        target_val = tf.squeeze(self.targetnet.predict(next_states)).numpy()

        erreur = np.zeros(self.batchsize)
        for i in range(self.batchsize):
            val = target[i][actions[i]]
            if dones[i]: target[i][actions[i]] = rewards[i]
            else:target[i][actions[i]] = self.gamma * target_val[i][np.argmax(target_next[i])] + rewards[i]
            erreur[i] = abs(val - target[i][actions[i]])
        
        for i in range(self.batch_size):
            idx = experiences[i][0]
            self.updatememory(idx, erreur[i])
    
        self.policynet.fit(states, tf.constant(target), batch_size=self.batch_size,sample_weight=None,verbose=0, epochs=3)
    
    def get_explo_rate(self):
        return self.end + np.exp(-1. * self.decay * self.current_step) * (self.start - self.end)

    def addmemory(self, experience):
        state, action, reward , next_state, done = utilities.extract_tensors([experience])

        target = self.policynet(state).numpy()
        target_next = self.policynet(next_state).numpy()
        target_val = self.targetnet(next_state).numpy()

        val = target[0][action[0]]
        if done[0]: target[0][action[0]] = reward[0]
        else:target[0][action[0]] = self.gamma * target_val[0][np.argmax(target_next[0])] + reward[0]
        erreur = abs(val - target[0][action[0]])
        self.tree.add((erreur + self.e) ** self.a, experience)

    def samplememory(self, n):
        batch = []
        moyenne = self.tree.tree[0] / n
        for i in range(n):
            a = moyenne * i
            b = moyenne * (i + 1)
            idx, p, data = self.tree.get(np.random.uniform(a, b))
            batch.append([idx, data])
        return batch

    def updatememory(self, idx, error):
        self.tree.update(idx, (error + self.e) ** self.a)

    def select_action(self,state):
        self.current_step +=1
        if self.episode >= self.supervised : 
            if self.get_explo_rate(self.current_step) > np.random.random():
                return np.random.randrange(self.num_actions)
            else:
                return ((tf.squeeze(tf.math.argmax(self.policynet.model(tf.expand_dims(state,0)),axis=1))).numpy())
        else: return(self.supervise())

    def load(self,name,dir):
        print(f"------------CHARGEMENT MODELE {name}------------")
        try:
            if name =="Target":self.targetnet = tf.keras.models.load_model(dir)
            else : self.policynet = tf.keras.models.load_model(dir)
            print(f"--------------MODELE CHARGE {name}--------------")
        except Exception as ex: print("échec du chargement : ",ex)

    def save(self,name):
        print(f"----------ENREGISTREMENT MODELE {name}----------")
        try: 
            if name =="Target":self.targetnet.save(self.saveplace2) 
            else : self.policynet.save(self.saveplace1) 
            print(f"------------MODELE ENREGISTRE {name}------------")
        except Exception as ex: print("échec de l'enregistrement : ",ex)
    
    def reset(self):
        pass

    def supervise(self):
        pass

    def main_loop(self):
        for episode in range(self.episodes):
            self.env.reset()
            state = self.env.get_state()
            for pas in count():
                action = self.select_action(state)
                reward,done = self.env.take_action(action)
                next_state = self.env.get_state()
                self.memory.add(utilities.Experience(state,action,reward,next_state,done),self.policynet.model,self.targetnet.model,self.gamma)
                state = next_state
                self.fitting()
                    
                if self.env.done:
                    self.tautransfer()
                    break
                if not self.tau_update:
                    if episode%self.target_update==0:self.transfer()
        
        if self.save:self.policynet.save()
