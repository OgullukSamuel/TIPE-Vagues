import numpy as np
import tensorflow as tf
import simu_for_PPO
import tensorflow_probability as tfp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
from collections import deque

class PPOMemory:
    """
    on gère la mémoire des actions pendant un episode
    """
    def __init__(self, batch_size,memory_size):
        self.clear_memory()
        self.batch_size = batch_size
        self.memory_size=memory_size
        self.moyenne, self.stddev = self.memory_size//5, self.memory_size//5
        self.deleted_idx=0


    def generate_batches(self):
        batch_start = np.arange(0, self.cardinal, self.batch_size)
        indices = np.arange(self.cardinal)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return( np.array(self.states),np.array(self.actions),np.array(self.logprobs),np.array(self.vals),np.array(self.rewards),np.array(self.dones),batches)

    def store_memory(self, state, action, probs, vals, reward, done):
        self.cardinal +=1
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        if self.cardinal >=self.memory_size:
            idx = int(np.random.normal(self.moyenne, self.stddev, 1))
            self.states.pop(idx)
            self.actions.pop(idx)
            self.logprobs.pop(idx)
            self.vals.pop(idx)
            self.rewards.pop(idx)
            self.dones.pop(idx)

    def clear_memory(self):
        self.cardinal=0
        self.states = []
        self.logprobs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []



class Agent(tf.Module):  
    """
    Agent utilisant du DQL 
    """
    def __init__(self,env):
        ##################################################################                          
        LearningRate = 1e-3
        self.env=env
        self.inputshape1=env.inputshape1
        self.inputshape2=env.inputshape2
        self.action_space=env.action_space	
        self.supervised=0                            #nb de tours en supervisés
        self.actor_opti=tf.keras.optimizers.Adam(learning_rate=LearningRate)
        self.critic_opti=tf.keras.optimizers.Adam(learning_rate=LearningRate)
        self.dir="model"                        # fichier où j'enregistre le modèle
        self.gamma=0.999                       	# taux de diminution de la récompense
        self.lmbda=0.95
        self.clip=0.2
        self.episodes = 50
        self.epochs=8                   	    
        self.fitting_steps=30                          #nombre d'étape entre chaque fit

        self.batch_size=20                             # taille du batch de Replay memory sur lequel j'entraine le modèle à chaque tour
        self.checkpoint=20                          #episodes avant un checkpoint
        self.memory_size=10000
        self.episode=3
        self.saving = True

        self.shuffle=True
        self.entropy_loss=0.001                     # augmenter si on veux un agent explorateur
        ###################################################################
        #création des RN et sauvegarde
        self.summon_nets()
        self.action_mem=deque()
        self.reward_mem=deque()
        self.vals_mem=deque()
        self.memory = PPOMemory(self.batch_size,self.memory_size)
        if not os.path.exists(self.dir): os.makedirs(self.dir)
        #self.save("actor",0)
        #self.save("critic",0)

        print("====== initialisation correcte ======")

    def summon_nets(self):
        """
        on créé le réseau neuronal
        """
        input1=tf.keras.layers.Input(shape=(16),name="Input_state")
        input2=tf.keras.layers.Input(shape=(16),name="Input_infos")
        
        dense1=tf.keras.layers.Dense(16,activation="relu",name="biais")(input1)
        dense2=tf.keras.layers.Dense(self.inputshape2,name="homogeneisation_dimensionnelle")(dense1)

        #attention=tf.keras.layers.Attention(name="scaled_dot_product_attention")([dense2,input2])     # normalement c'est une attention layer
        
        value_net = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(1,activation="tanh"),        
            tf.keras.layers.Lambda(lambda s: tf.keras.backend.expand_dims(s[:, 0], -1),output_shape=(self.action_space,))],name="Value_network")(dense2)

        advantage_net = tf.keras.Sequential([ 
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(self.action_space),
            tf.keras.layers.Lambda(lambda a: a[:, :] - tf.keras.backend.mean(a[:, :], keepdims=True),output_shape=(self.action_space,))],name="Advantage_network")(dense2)

        add = tf.keras.layers.Add(name="sommation")([value_net, advantage_net])
        
        #normalisation = tf.keras.layers.Dense(self.action_space,name="normalisation")(dense2)#(add)        
        soft= tf.keras.layers.Dense(self.action_space,activation="softmax")(add)


        out_critic=tf.keras.layers.Dense(16,name="normalisation") (add)
        output_critic = tf.keras.layers.Dense(1,activation="tanh")(out_critic)

        actor = tf.keras.Model(input1, soft,name="actor")
        critic= tf.keras.Model(input1, output_critic,name="critic")
        actor.compile(optimizer=self.actor_opti)
        critic.compile(optimizer=self.critic_opti)
        actor.summary()
        #tf.keras.utils.plot_model(actor, to_file='réseau.png')
        self.actornet=actor
        self.criticnet=critic


    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def plotting_memory(self,deque):
        """
        je gère l'enregistrement dans la mémoire pour pouvoir plot après
        """
        #print(deque)
        n=len(deque)
        actions=np.zeros(n,dtype="int16")
        rewards=np.zeros(n)
        vals=np.zeros(n)
        temp=np.zeros((n,self.action_space))
        for i in range(n): 
            rewards[i],vals[i],actions[i],vals[i] = deque.pop()
            _,temp[i] = np.unique(np.concatenate((np.arange(0,self.action_space),[actions[i]])),return_counts=True)
        
        self.action_mem.append(list(np.mean(temp-1,axis=0)))
        self.reward_mem.append([min(rewards),np.mean(rewards),max(rewards)])
        self.vals_mem.append(np.mean(vals))
        


    def training(self):
        best_score = 100
        score_history = []
        learn_iters = 0
        avg_score = 0
        n_steps = 0
        k=0
        for i in range(self.episodes):
            state = env.reset()
            done = 0
            score = 0
            memdeque=deque()
            while not done:
                action, prob, val = self.choose_action(state)
                next_state, reward, done= env.step(tf.keras.utils.to_categorical(action, num_classes=env.action_space) )
                n_steps += 1
                score += reward
                memdeque.append([reward,prob,action,val[0]])
                self.store_transition(state, action,prob, val, reward, done)
                if n_steps % self.fitting_steps == 0:
                    self.learn()
                    learn_iters += 1
                state = next_state
            score_history.append(score)
            self.plotting_memory(memdeque)

            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                k+=1
                self.save("actor",k)
                self.save("critic",k)
            print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,'time_steps', n_steps, 'learning_steps', learn_iters)
        self.plotting()

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        probs = self.actornet(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).numpy()[0]
        value = self.criticnet(state).numpy()[0]
        return(action.numpy()[0], log_prob, value)

    def GAE(self,rew,done,val):
        
        advantage = np.zeros(len(rew))
        for t in range(len(rew)-1):
            discount = 1
            sum = 0
            for k in range(t, len(rew)-1):
                sum += discount*(rew[k] + self.gamma*val[k+1]*(1-done[k]) - val[k])
                discount *= self.gamma*self.lmbda
            advantage[t] = sum
        return(advantage)

    def learn(self):
        for _ in range(self.epochs):
            etats, acts, old_probas, values,rewards, dones, batches = self.memory.generate_batches()

            advantage =self.GAE(rewards,dones,values)

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(etats[batch])
                    old_probs = tf.convert_to_tensor(old_probas[batch])
                    actions = tf.convert_to_tensor(acts[batch])
                    probs = self.actornet(states)
                    new_probs = tfp.distributions.Categorical(probs).log_prob(actions)
                    critic_value = tf.squeeze(self.criticnet(states),1)
                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    clipped_probs = tf.clip_by_value(prob_ratio,1-self.clip,1+self.clip)
                    actor_loss = tf.math.reduce_mean(-tf.math.minimum(advantage[batch] * prob_ratio,clipped_probs * advantage[batch]))
                    critic_loss = tf.keras.losses.MSE(critic_value, advantage[batch] + values[batch])

                actor_params = self.actornet.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.criticnet.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.actornet.optimizer.apply_gradients(zip(actor_grads, actor_params))
                self.criticnet.optimizer.apply_gradients(zip(critic_grads, critic_params))
        self.memory.clear_memory()
    
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
    
    def plotting(self):
        loutre = np.transpose(self.reward_mem)
        print(self.vals_mem)
        x=np.arange(0,self.episodes)
        fig = plt.figure()
        
        ax = fig.add_subplot(2,1,1)
        ax.plot(x,loutre[0])
        ax.plot(x,loutre[1])
        ax.plot(x,loutre[2])
        ax.set_ylabel('recompense')
        ax.set_xlabel('episodes')


        
        ax = fig.add_subplot(223, projection='3d')
        y=np.array(self.action_mem)
        _xx, _yy = np.meshgrid(np.arange(self.action_space), np.arange(self.episodes))
        a, b = _xx.ravel(), _yy.ravel()

        top = y.flatten()
        bottom = np.zeros_like(top)
        width = depth = 1
        ax.bar3d(a, b, bottom, width, depth, top, shade=True)
        ax.set_title('actions')
        ax.set_ylabel('actions')
        ax.set_xlabel('episodes')
        fig.align_labels()
        
        ax = fig.add_subplot(224)
        ax.plot(x,self.vals_mem)
        ax.set_ylabel('avg_critic_predictions')
        ax.set_xlabel('episodes')
        fig.align_labels()

        plt.show()





if __name__ == '__main__':
    env=simu_for_PPO.Environnement()
    herbert=Agent(env)
    herbert.training()