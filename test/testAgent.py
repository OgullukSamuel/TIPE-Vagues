import tensorflow as tf
import numpy as np
import os
import h5py
import utilities
import parametres
import environnement

print("GPUs : ", tf.config.list_physical_devices('GPU'))

class Agent(tf.Module):  
    """
    création d'un réseau de neurone profond 
    """
    def __init__(self,env):
        param=parametres.para()
        self.env=env;self.tau_update=param.tau_update
        self.dueling=param.dueling;self.TAU = param.tau
        self.inputshape=param.inputshape;self.actionspace=param.actionspace
        self.initializer=param.initializer;self.supervised=param.supervised
        self.opti=param.opti;self.lossing=param.loss
        self.dir=param.dir;self.metrique=param.metrique
        self.tau_update=param.tau_update;self.gamma=param.gamma
        self.episodes =param.episodes;self.epochs=param.epochs
        self.batch_size=param.batch_size;self.memory_size=param.memory_size
        self.a=param.PER_alpha;self.e=param.PER_epsi
        self.batchsize=param.batch_size
        self.tree = utilities.SumTree(self.memory_size)
        self.start = param.eps_st;self.end = param.eps_end								
        self.decay = param.eps_decay;self.num_actions = param.actionspace
        self.target_update=param.target_update;self.checkpoint=param.checkpoint
        self.episode=0

        self.summon_net("Policy")
        self.summon_net("Target")

        if not os.path.exists(self.dir): os.makedirs(self.dir)
        self.saveplace1 = os.path.join(self.dir,"Policy_network"+".h5")
        self.saveplace2 = os.path.join(self.dir,"Target_network"+".h5") 
        print("====== initialisation correcte ======")

    def summon_net(self,nom):
        input1=tf.keras.layers.Input(shape=self.inputshape,name="Input_gradient",batch_size=self.batch_size)
        input2=tf.keras.layers.Input(shape=self.inputshape,name="Input_context")

        convo_grad = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=3,kernel_size=2, strides=2,input_shape=self.inputshape[1:]),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding="valid"),
            tf.keras.layers.Conv2D(filters=2,kernel_size=2, strides=2),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")],name="Convolution_gradient")(input1)

        convo_context= tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=3,kernel_size=2, strides=2,input_shape=self.inputshape[1:]),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding="valid"),
            tf.keras.layers.Conv2D(filters=2,kernel_size=2, strides=2),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")],name="Convolution_context")(input2)

        merge = tf.keras.layers.Multiply(name="produit_hadamard")([convo_grad,convo_context])
        reshape=tf.keras.layers.Reshape((178,159),name="reshape")(merge)
        patrick_GRUel=tf.keras.layers.GRU(16,name="RN_recurrent")(reshape) #,stateful=True

        if self.dueling:
            value_net = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation="relu", kernel_initializer=self.initializer),
                tf.keras.layers.Dense(1, kernel_initializer=self.initializer),
                tf.keras.layers.Lambda(lambda s: tf.keras.backend.expand_dims(s[:, 0], -1),output_shape=(self.actionspace,))],name="Value_network")(patrick_GRUel)

            advantage_net = tf.keras.Sequential([ 
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu", kernel_initializer=self.initializer),
                tf.keras.layers.Dense(self.actionspace, kernel_initializer=self.initializer),
                tf.keras.layers.Lambda(lambda a: a[:, :] - tf.keras.backend.mean(a[:, :], keepdims=True),output_shape=(self.actionspace,))],name="Advantage_network")(patrick_GRUel)

            add = tf.keras.layers.Add(name="sommation")([value_net, advantage_net])
            output = tf.keras.layers.Dense(self.actionspace, kernel_initializer=self.initializer,activation="softmax",name="normalisation")(add)
        else: output = tf.keras.layers.Dense(self.actionspace, kernel_initializer=self.initializer,activation="softmax",name="normalisation")(patrick_GRUel)

        model = tf.keras.Model([input1,input2], output,name=nom)

        model.compile(optimizer=self.opti, loss=self.lossing, metrics=self.metrique)
        #tf.keras.utils.plot_model(model, to_file='réseau.png')
        model.summary()
        if nom=="Target": self.targetnet=model
        else:self.policynet=model

    def tautransfer(self):
        if not self.tau_update and self.episode%self.target_update==0:
            self.transfer()
        else:
            q_model_theta = self.policynet.get_weights()
            target_model_theta = self.targetnet.get_weights()
            i = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                target_model_theta[i] = target_weight
                i += 1
            self.targetnet.set_weights(target_model_theta)

    def transfer(self):
        self.targetnet.set_weights(self.policynet.get_weights())

    def fitting(self):
        experiences = self.samplememory(self.batchsize)
        states, actions, rewards , next_states, dones =  utilities.extract_tensors(list(zip(*experiences))[1])
        target = self.policynet.predict([states[:][0],states[:][1]])

        target_next = self.policynet.predict([next_states[:][0],next_states[:][1]])
        target_val = self.targetnet.predict([next_states[:][0],next_states[:][1]])

        erreur = np.zeros(self.batchsize)
        for i in range(self.batchsize):
            val = target[i][actions[i]]
            if dones[i]: 
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = self.gamma * target_val[i][np.argmax(target_next[i])] + rewards[i]
            erreur[i] = np.abs(val - target[i][actions[i]])
        
        for i in range(self.batch_size):
            self.updatememory(experiences[i][0], erreur[i])
    
        self.policynet.fit([states[:][0],states[:][1]], target, batch_size=self.batch_size,sample_weight=None,verbose=0, epochs=self.epochs)
    
    def get_explo_rate(self):
        return self.end + np.exp(-1. * self.decay * self.current_step) * (self.start - self.end)

    def addmemory(self, experience):
        state, action, reward , next_state, done =  utilities.extract_tensors([experience])

        target = self.policynet.predict_on_batch([np.expand_dims(state[0],axis=0),np.expand_dims(state[1],axis=0)])[0]
        target_next = self.policynet([np.expand_dims(next_state[0],axis=0),np.expand_dims(next_state[1],axis=0)])[0]
        target_val = self.targetnet.predict_on_batch([np.expand_dims(next_state[0],axis=0),np.expand_dims(next_state[1],axis=0)])[0]

        val = target[action[0]]
        if done[0]: target[action[0]] = reward[0]
        else:target[action[0]] = self.gamma * target_val[np.argmax(target_next[0])] + reward[0]
        erreur = np.abs(val - target[action[0]])
        self.tree.add(np.power(erreur + self.e, self.a), experience)

    def samplememory(self, n):
        batch = []
        moyenne = self.tree.tree[0] / n
        for i in range(n):
            batch.append(self.tree.get(np.random.uniform(moyenne * i, moyenne * (i + 1))))
        return batch

    def updatememory(self, idx, error):
        self.tree.update(idx, np.power(error + self.e, self.a))

    def select_action(self,state):
        self.current_step +=1
        if self.episode >= self.supervised : 
            if self.get_explo_rate() > np.random.random():
                return np.random.randint(0,self.actionspace)
            else:
                return tf.squeeze(tf.argmax(tf.squeeze(self.policynet(state)))).numpy()
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
        self.env.reset()
        self.current_step=0
        self.policynet.reset_states()


    def supervise(self):
        pass

    def main_loop(self):
        for _ in range(self.episodes):
            self.episode+=1
            self.reset()
            state = self.env.get_state()
            while not self.env.done:
                action = self.select_action(state)
                reward,done = self.env.take_action(action)
                next_state = self.env.get_state()
                self.addmemory(utilities.Experience(state,action,reward,next_state,done))
                state = next_state
                self.fitting()
                    
            self.tautransfer()
        
        if self.save and self.episode==self.checkpoint:self.policynet.save()


envs=environnement.ENV()
kevin = Agent(envs)
kevin.main_loop()