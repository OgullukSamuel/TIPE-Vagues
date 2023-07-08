import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import parametresPPO
import environnement
import matplotlib.pyplot as plt

class Agent(tf.Module):  
    """
    Agent utilisant du DQL 
    """
    def __init__(self,env):
        #initialisation des variables en utilisant param
        ##################################################################
        param=parametresPPO.para()                             
        self.env=env
        self.inputshape=param.inputshape
        self.action_space=env.action_space
        self.initializer=param.initializer;self.supervised=param.supervised
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
    

    #def summon_net(self,nom):
        """
        fonction de création des réseaux de neurones
        """
        """input1=tf.keras.layers.Input(shape=self.inputshape[1:],name="Input_state",batch_size=self.batch_size)
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

        model = tf.keras.Model([input1,input2,input3], output,name=nom)

        model.compile(optimizer=self.opti, loss=self.lossing, metrics=self.metrique)
        tf.keras.utils.plot_model(model, to_file='réseau.png')
        model.summary()
        if nom=="critic": self.criticnet=model
        else:self.actornet=model"""
    def summon_nets(self):
        input_actor=tf.keras.layers.Input(shape=self.inputshape[1:],name="Input_state",batch_size=self.batch_size)
        input_critic=tf.keras.layers.Input(shape=self.inputshape[1:],name="Input_tate",batch_size=self.batch_size)
        actor=tf.keras.Sequential([
            tf.keras.layers.Dense(64),
            #tf.keras.layers.Dropout(.1),
            tf.keras.layers.Reshape((1,64,)),
            tf.keras.layers.GRU(16),
            tf.keras.layers.Dense(self.action_space,activation="softmax")])(input_actor)

        critic=tf.keras.Sequential([
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dropout(.1),
            tf.keras.layers.Reshape((1,64,)),
            tf.keras.layers.GRU(16),
            tf.keras.layers.Dense(1)])(input_critic)

        actor_net = tf.keras.Model(input_actor, actor,name="actor")
        critic_net= tf.keras.Model(input_actor, critic,name="critic")
        actor_net.compile(optimizer=self.actor_opti, loss=self.actor_lossing, metrics=self.actor_metrique)
        critic_net.compile(optimizer=self.critic_opti, loss=self.critic_lossing, metrics=self.critic_metrique)
        self.actornet=actor_net
        self.criticnet=critic_net
        actor_net.summary()
        critic_net.summary()


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

        if self.dueling:
            value_net = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation="relu", kernel_initializer=self.initializer),
                tf.keras.layers.Dense(1, kernel_initializer=self.initializer),
                tf.keras.layers.Lambda(lambda s: tf.keras.backend.expand_dims(s[:, 0], -1),output_shape=(self.action_space,))],name="Value_network")(patrick_GRUel)

            advantage_net = tf.keras.Sequential([ 
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu", kernel_initializer=self.initializer),
                tf.keras.layers.Dense(self.action_space, kernel_initializer=self.initializer),
                tf.keras.layers.Lambda(lambda a: a[:, :] - tf.keras.backend.mean(a[:, :], keepdims=True),output_shape=(self.actionspace,))],name="Advantage_network")(patrick_GRUel)

            add = tf.keras.layers.Add(name="sommation")([value_net, advantage_net])
            output = tf.keras.layers.Dense(self.action_space, kernel_initializer=self.initializer,activation="softmax",name="normalisation")(add)
        else: output = tf.keras.layers.Dense(self.action_space, kernel_initializer=self.initializer,activation="softmax",name="normalisation")(patrick_GRUel)

        model = tf.keras.Model([input1,input2,input3], output,name=nom)

        model.compile(optimizer=self.opti, loss=self.lossing, metrics=self.metrique)
        tf.keras.utils.plot_model(model, to_file='réseau.png')
        model.summary()
        if nom=="Target": self.targetnet=model
        else:self.policynet=model



    def select_action(self,state):                                                                      
        action = tfp.distributions.Categorical(probs=self.actornet(np.array([state])),dtype=tf.float32).sample()
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
                proba = self.actornet(np.array(states), training=True)
                value =  self.criticnet(np.array(states),training=True)
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


    def training(self):
        for episode in range(self.episodes):
            done = False
            state = self.env.reset()
            probs = []
            states=[]
            rewardes=[]
            dones=[]
            actions=[]
            values = []
            while not done:  # boucle main
                action = self.select_action(state)
                value = self.criticnet(np.array([state])).numpy()
                nxt_state,reward,done=self.env.take_action(action)
                states.append(state.tolist())
                rewardes.append(reward)
                dones.append(1-done)
                actions.append(action)
                probs.append(self.actornet(np.array([state]))[0])
                values.append(value[0][0])
                state = nxt_state
                
            self.rewards.append(np.sum(rewardes))
            values.append(self.criticnet(np.array([state])).numpy()[0][0])


            print(f"episode : {episode} || modele numero : {episode//self.checkpoint+1} || dernière récompense : {self.rewards[-1]}")
            returns, adv = self.GAE(rewardes,dones, values)
            self.fitting(states, actions, probs, adv,  returns)  
            
            if episode % self.checkpoint ==0 and self.saving: 
                self.save(episode//self.checkpoint+1)    #on enregistre périodiquement les modèles par sécurité 


env=environnementPPO.ENV()

agentoo7 = Agent(env)
agentoo7.training()
env.close()
plt.plot(agentoo7.rewards)
plt.show()