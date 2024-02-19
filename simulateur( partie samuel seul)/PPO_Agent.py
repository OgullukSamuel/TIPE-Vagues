import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import matplotlib.pyplot as plt
import simu_for_PPO


class Agent(tf.Module):  
    """
    Agent utilisant du DQL 
    """
    def __init__(self,env):
        #initialisation des variables en utilisant param
        ##################################################################                          
        LearningRate = 1e-3
        self.env=env
        self.dueling=True                   # True si mise en place de double dueling networks
        self.inputshape1=env.inputshape1
        self.inputshape2=env.inputshape2
        self.action_space=env.action_space
        #self.initializer=tf.keras.initializers.RandomNormal(stddev=0.25,mean=0.5)	
        self.supervised=0                            #nb de tours en supervisés
        self.actor_opti=tf.keras.optimizers.Adam(learning_rate=LearningRate)
        self.critic_opti=tf.keras.optimizers.Adam(learning_rate=LearningRate)
        self.actor_metrique=[tf.keras.metrics.MeanSquaredError()] 
        self.critic_metrique=[tf.keras.metrics.MeanSquaredError()]   
        self.actor_lossing=tf.keras.losses.CategoricalCrossentropy() 
        self.critic_lossing=tf.keras.losses.MeanSquaredError()
        self.dir="model"                        # fichier où j'enregistre le modèle
        self.gamma=0.999                       	# taux de diminution de la récompense
        self.lmbda=0.95
        self.clip=0.2
        self.episodes = 2
        self.epochs=15
        self.batch_size=2                     	    
        self.batchsize_fit= 2                      	# taille du batch avec lequel j'utilise la fonction fit

        self.batch_size=2                               # taille du batch de Replay memory sur lequel j'entraine le modèle à chaque tour
        self.checkpoint=20                          #episodes avant un checkpoint
        self.episode=3
        self.saving = True
        self.rewards=[]
        self.episode_duration=50
        ###################################################################
        #création des RN et sauvegarde
        self.summon_nets()
        if not os.path.exists(self.dir): os.makedirs(self.dir)
        #self.save("actor",0)
        #self.save("critic",0)

        print("====== initialisation correcte ======")

    def reset(self):
        # on reset tout l'environnement et la mémoire des RN 
        self.env.reset()
        self.current_step=0
        self.actornet.reset_states()
        self.criticnet.reset_states()
        
    
    def load(self,name,dir):
        """
        simple fonction utilitaire pour charger facilement un RN 
        """
        print(f"------------CHARGEMENT MODELE {name}------------")
        try:
            if name =="actor":self.actornet = tf.keras.models.load_model(dir)
            else : self.criticnet = tf.keras.models.load_model(dir)
            print(f"--------------MODELE CHARGE {name}--------------")
        except Exception as ex: print("échec du chargement : ",ex)

    def save(self,name,nb):
        """
        fonction pour sauvegarder les RN 
        """
        print(f"----------ENREGISTREMENT MODELE {name}----------")
        try:
            saveplace = os.path.join(self.dir, f"{name}{nb}.keras") 
            if name =="actor": self.actornet.save(saveplace) 
            else : self.criticnet.save(saveplace)

            print(f"------------MODELE ENREGISTRE {name}------------")
        except Exception as ex: print("échec de l'enregistrement : ",ex)
    
    def summon_nets(self):
        """
        on créé le réseau neuronal
        """
        input1=tf.keras.layers.Input(shape=self.inputshape1,name="Input_state",batch_size=self.batch_size)
        input2=tf.keras.layers.Input(shape=self.inputshape2,name="Input_infos",batch_size=self.batch_size)

        dense1=tf.keras.layers.Dense(self.inputshape1[0],name="biais")(input1)
        flat=tf.keras.layers.Flatten()(dense1)
        dense2=tf.keras.layers.Dense(self.inputshape2,name="homogeneisation_dimensionnelle")(flat)

        attention=tf.keras.layers.Attention(name="scaled_dot_product_attention")([dense2,input2])

        value_net = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda s: tf.keras.backend.expand_dims(s[:, 0], -1),output_shape=(self.action_space,))],name="Value_network")(attention)

        advantage_net = tf.keras.Sequential([ 
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(self.action_space),
            tf.keras.layers.Lambda(lambda a: a[:, :] - tf.keras.backend.mean(a[:, :], keepdims=True),output_shape=(self.action_space,))],name="Advantage_network")(attention)

        add = tf.keras.layers.Add(name="sommation")([value_net, advantage_net])
        output_actor = tf.keras.layers.Dense(self.action_space,activation="tanh",name="normalisation")(add)

        out_critic=tf.keras.layers.Dense(32,name="normalisation") (attention)
        output_critic = tf.keras.layers.Dense(1, activation="sigmoid")(out_critic)

        actor = tf.keras.Model([input1,input2], output_actor,name="actor")
        critic= tf.keras.Model([input1,input2], output_critic,name="critic")
        actor.compile(optimizer=self.actor_opti, loss=self.actor_lossing, metrics=self.actor_metrique)
        critic.compile(optimizer=self.critic_opti, loss=self.critic_lossing, metrics=self.critic_metrique)
        actor.summary()
        #tf.keras.utils.plot_model(actor, to_file='réseau.png')
        self.actornet=actor
        self.criticnet=critic

    def select_action(self,state): 
        """
        on choisit l'action à effectuer aléatoirement selon une distribution donnée par notre réseau neuronal
        """                               

        action = tfp.distributions.Categorical(probs=self.actornet(state)).sample(2)
        return np.squeeze(action.numpy())


    def GAE(self,rewards, dones,values):
        n,g=len(rewards),0
        returns = np.zeros(n)
        for i in reversed(range(n)):
            delta = rewards[i] + self.gamma * values[i + 1] * dones[i] - values[i]
            g = delta + self.gamma * self.lmbda * dones[i] * g
            returns[i]= g + values[i]

        adv = returns.copy() - values[:-1]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        return  returns, adv    

    def fitting(self, states, actions,  adv , old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = tf.squeeze([self.actornet(states[i], training=True) for i in range(len(states))])
            v =  tf.squeeze([self.criticnet(states[i],training=True) for i in range(len(states))])
            v = tf.reshape(v, (len(v),))
            c_loss = 0.5 * tf.keras.losses.mean_squared_error(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)

        grads1 = tape1.gradient(a_loss, self.actornet.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.criticnet.trainable_variables)
        self.actor_opti.apply_gradients(zip(grads1, self.actornet.trainable_variables))
        self.critic_opti.apply_gradients(zip(grads2, self.criticnet.trainable_variables))
        return a_loss, c_loss

    def actor_loss(self, probs, actions, adv, old_probs, closs):
        probability = probs      
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        sur1 = []
        sur2 = []
        for pb, t, op, a in zip(probability, adv, old_probs, actions):
            t =  tf.cast(t,dtype=tf.float32)
            ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))

            sur1.append(tf.math.multiply(ratio,t))
            sur2.append(tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip, 1.0 + self.clip),t))
        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        return loss


    def training(self):
        rewards=[]
        for episode in range(self.episodes):
            
            done = False
            state = self.env.reset()
            probs = []
            states=[]
            rewardes=[]
            dones=[]
            actions=[]
            values = []
            ep=0
            while not done:  # boucle main
                ep+=1
                action = self.select_action(state)
                value = self.criticnet(state).numpy()
                nxt_state, reward, done = self.env.step(action)
                states.append(list(state))
                rewardes.append(reward)
                dones.append(1-done)
                actions.append(action)
                probs.append(self.actornet(state)[0])
                values.append(value[0][0])
                state = nxt_state
                if ep>self.episode_duration:done=True
                
            rewards.append(np.sum(rewardes))
            values.append(self.criticnet(state).numpy()[0][0])

            returns, adv = self.GAE(rewardes,dones, values)
            print(f"episode : {episode} || modele numero : {episode//self.checkpoint+1} || dernière récompense : {rewards[-1]}")
            for _ in range(self.epochs):
                al,cl = self.fitting(states, actions, adv, probs, returns)  
            
            if episode % self.checkpoint ==0 and self.saving: 
                self.save("actor",episode//self.checkpoint+1)    #on enregistre périodiquement les modèles par sécurité 
                self.save("critic",episode//self.checkpoint+1)   
        self.rewards=rewards


env=simu_for_PPO.Environnement()
#print(env.get_state()[0])
berthelot = Agent(env)
berthelot.training()
#plt.plot(agentoo7.rewards)
#plt.show()
