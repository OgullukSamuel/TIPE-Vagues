import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import utilities
import parametresPPO
import environnement

tf.random.set_seed(1234)

import matplotlib.pyplot as plt
class Agent(tf.Module):  
    """
    Agent utilisant la PPO ( Post proximal policy)
    """
    def __init__(self,env):
        #initialisation des variables en utilisant param
        ##################################################################
        
        param=parametresPPO.para()                             
        self.env=env
        self.inputshape=param.inputshape
        self.action_space=env.action_space
        self.supervised=param.supervised
        self.actor_opti=param.actor_opti
        self.critic_opti=param.critic_opti
        self.actor_metrique=param.actor_metrique
        self.critic_metrique=param.critic_metrique
        self.actor_lossing=param.actor_loss
        self.critic_lossing=param.critic_loss
        self.dir=param.dir
        self.gamma=param.gamma
        self.lmbda=param.lmbda
        self.clip=param.clip
        self.episodes =param.episodes
        self.epochs=param.epochs
        self.batch_size=param.batch_size
        self.checkpoint=param.checkpoint
        self.episode=0
        self.saving = param.save
        self.rewards=[]
        self.amplitude,self.episode_duration=param.amplitude,param.episode_duration
         
        ###################################################################
        #création des RN et sauvegarde
        self.summon_nets()
        if not os.path.exists(self.dir): os.makedirs(self.dir)
        self.save(0)


        print("====== initialisation correcte ======")

    def reset(self):
        # on reset tout l'environnement et la mémoire des RN 
        self.env.reset()
        self.current_step=0
        self.actornet.reset_states()
        self.criticnet.reset_states()
        
    
    def load(self,nb):
        """
        simple fonction utilitaire pour charger facilement un RN 
        """
        print("------------CHARGEMENT MODELE ------------")
        try:
            self.actornet = tf.keras.models.load_model(os.path.join(self.dir, f"actor_net{nb}.keras"))
            self.criticnet = tf.keras.models.load_model(os.path.join(self.dir, f"critic_net{nb}.keras"))
            print("--------------MODELE CHARGE --------------")
        except Exception as ex: print("échec du chargement : ",ex)

    def save(self,nb):
        """
        fonction pour sauvegarder les RN 
        """
        print("----------ENREGISTREMENT MODELES----------")
        try:
            self.actornet.save(os.path.join(self.dir, f"actor_net{nb}.keras") ) 
            self.criticnet.save(os.path.join(self.dir, f"critic_net{nb}.keras") )
            print("------------MODELES ENREGISTRES------------")
        except Exception as ex: print("échec de l'enregistrement : ",ex)
    

    def summon_net(self,name):
        """
        fonction de création des réseaux de neurones
        """
        input1=tf.keras.layers.Input(shape=self.inputshape[1:],name="Input_state",batch_size=self.batch_size)
        input2=tf.keras.layers.Input(shape=self.inputshape[1:],name="Input_context")
        input3=tf.keras.layers.Input(shape=((3,2)),name="Input_data")

        convo_grad = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=2,kernel_size=3, strides=2,input_shape=self.inputshape[1:]),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding="valid"),
            tf.keras.layers.Conv2D(filters=2,kernel_size=2, strides=2),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")],name="Convolution_state")(input1)

        convo_state= tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=2,kernel_size=3, strides=2,input_shape=self.inputshape[1:]),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding="valid"),
            tf.keras.layers.Conv2D(filters=2,kernel_size=2, strides=2),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")],name="Convolution_context")(input2)

        merge = tf.keras.layers.Multiply(name="produit_hadamard")([convo_grad,convo_state])
        convo_finale= tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=2,kernel_size=3, strides=3),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")],name="Convolution_merge")(merge)
        reshape=tf.keras.layers.Reshape((9*13,2),name="reshape")(convo_finale)
        attention=tf.keras.layers.Attention(name="scaled_dot_product_attention")([reshape,input3])
        patrick_GRUel=tf.keras.layers.GRU(8,name="RN_recurrent")(attention) #,stateful=True

        if name=="actor":
            value_net = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64,activation="relu"),
                tf.keras.layers.Dense(1),
                tf.keras.layers.Lambda(lambda s: tf.keras.backend.expand_dims(s[:, 0], -1),output_shape=(self.action_space,))],name="Value_network")(patrick_GRUel)

            advantage_net = tf.keras.Sequential([ 
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(self.action_space),
                tf.keras.layers.Lambda(lambda a: a[:, :] - tf.keras.backend.mean(a[:, :], keepdims=True),output_shape=(self.action_space,))],name="Advantage_network")(patrick_GRUel)

            add = tf.keras.layers.Add(name="sommation")([value_net, advantage_net])
            output = tf.keras.layers.Dense(self.action_space,activation="softmax",name="normalisation")(add)
        else:
            out=tf.keras.layers.Dense(32,name="normalisation") (patrick_GRUel)
            output = tf.keras.layers.Dense(1, activation="tanh")(out)

        model = tf.keras.Model([input1,input2,input3], output,name=name)

        tf.keras.utils.plot_model(model, to_file=f'{name}net.png')
        model.summary()
        if name=="actor":
            model.compile(optimizer=self.actor_opti, loss=self.actor_lossing, metrics=self.actor_metrique)
         
            self.actornet=model
        else:
            model.compile(optimizer=self.critic_opti, loss=self.critic_lossing, metrics=self.critic_metrique)
            self.criticnet=model

    def summon_nets(self):
        """
        critic = 1 scalaire
        actor = self.action_space
        """
        self.summon_net("actor")
        self.summon_net("critic")


    def select_action(self,state):                                                                      
        action = tfp.distributions.Categorical(probs=self.actornet([state]),dtype=tf.float32).sample()
        return int(action.numpy()[0])


    def GAE(self,rewards, dones,values):
        n,g=len(rewards),0
        returns = np.zeros(n)
        for i in reversed(range(n)):
            delta = rewards[i] + self.gamma * values[i + 1] * dones[i] - values[i]
            g = delta + self.gamma * self.lmbda * dones[i] * g
            returns[i]=g + values[i]

        adv = returns.copy() - values[:-1]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        return  returns, adv    

    def fitting(self, states, actions,  old_probs,adv ,  discnt_rewards):
        for _ in range(self.epochs):
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                #print(states)
                proba=[]
                value=[]
                for i in states:
                    proba.append(self.actornet(i,training=True)[0])    
                    value.append(self.criticnet(i,training=True)[0])
                #proba = self.actornet.predict_on_batch([[states]])
                #value =  self.criticnet.predict([states[:][0],states[:][1],states[:][2]])
                print(np.array(proba).shape,np.array(value).shape)
                c_loss = 0.5 * tf.keras.losses.mean_squared_error(discnt_rewards, value)
                a_loss = self.actor_loss(proba, actions, adv, old_probs, c_loss)

            grads1 = tape1.gradient(a_loss, self.actornet.trainable_variables)
            grads2 = tape2.gradient(c_loss, self.criticnet.trainable_variables)
            self.actor_opti.apply_gradients(zip(grads1, self.actornet.trainable_variables))
            self.critic_opti.apply_gradients(zip(grads2, self.criticnet.trainable_variables))

    def actor_loss(self, probs, actions, adv, old_probs, closs):
        probability = probs      
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        sur1 = []
        sur2 = []
        for pb, t, op, a in zip(probability, adv, old_probs, actions):
            t =  tf.cast(t,dtype=tf.float32)
            ratio = tf.math.divide(pb[a],op[a])

            sur1.append(tf.math.multiply(ratio,t))
            sur2.append(tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip, 1.0 + self.clip),t))
        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        return loss

    def ep_run(self):
        done = False
        self.probs = []
        self.states=[]
        self.rewardes=[]
        self.dones=[]
        self.actions=[]
        self.values = []
        k=1
        while not done:  # boucle main
            self.env.move(k,self.amplitude,timeless=True)
            k*=-1
            action =self.select_action(self.state)
            value = self.criticnet(self.state).numpy()
            nxt_state,reward,done=self.env.take_action(action)
            self.states.append([self.state])
            self.rewardes.append(reward)
            self.dones.append(1-done)
            self.actions.append(action)
            self.probs.append(self.actornet([self.state]))
            self.values.append(value[0][0])
            self.state = nxt_state 

    def reload_exp(self):
        with open('test.npy', 'rb') as f:
            a = np.load(f)
        self.rewards=a[1]
        self.load(a[2])
        self.training(a[0])

    def training(self,retake=0):
        for episode in range(retake,self.episodes):
            self.state = self.env.reset()
            self.ep_run()
                
            self.rewards.append(np.sum(self.rewardes))
            self.values.append(self.criticnet([self.state]).numpy()[0][0])


            print(f"episode : {episode} || modele numero : {episode//self.checkpoint+1} || dernière récompense : {self.rewards[-1]}")
            returns, adv = self.GAE(self.rewardes.copy(),self.dones.copy(), self.values.copy())
            self.fitting(self.states.copy(), self.actions.copy(), self.probs.copy(), adv,  returns)  
            
            if episode % self.checkpoint ==0 and self.saving: 
                self.save(episode//self.checkpoint+1)    #on enregistre périodiquement les modèles par sécurité 
    
            with open('data.npy', 'wb') as f:
                np.save(f, np.array([episode,self.rewards[-1],episode//self.checkpoint+1,self.dones]))
        plt.plot(self.rewards)
        plt.show()
env=environnement.ENV()
print("poulet1")
bernard = Agent(env)
print("poulet")
bernard.training()
plt.plot(bernard.rewards)
plt.show()