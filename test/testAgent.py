import tensorflow as tf
import numpy as np
import os
import utilities
import parametres
import environnement
import sqlite3

a=tf.config.list_physical_devices('GPU')
print("nombres de GPU disponibles : ", len(a))
if(len(a)>0):print("GPU disponibles :",a)

class Agent(tf.Module):  
    """
    Agent utilisant du DQL 
    """
    def __init__(self,env):
        #initialisation des variables en utilisant param
        ##################################################################
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
        self.batchsize_fit=param.batchsize_fit
        self.batch_size=param.batch_size
        self.tree = utilities.SumTree(self.memory_size)
        self.start = param.eps_st;self.end = param.eps_end								
        self.decay = param.eps_decay;self.num_actions = param.actionspace
        self.target_update=param.target_update;self.checkpoint=param.checkpoint
        self.episode=0
        self.saving = param.save
        self.rewards=np.zeros(self.episodes)
        ###################################################################
        #création des RN et sauvegarde
        self.summon_net("Policy")
        self.summon_net("Target")
        if not os.path.exists(self.dir): os.makedirs(self.dir)
        self.save("Policy",0)
        self.save("Target",0)

        
        print("====== initialisation correcte ======")

    def summon_net(self,nom):
        """
        fonction de création des réseaux de neurones
        """
        input1=tf.keras.layers.Input(shape=self.inputshape[1:],name="Input_gradient",batch_size=self.batch_size)
        input2=tf.keras.layers.Input(shape=self.inputshape[1:],name="Input_context")

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
        """
        fonction effectuant le transfert des poids du réseau policy à Target
        selon deux méthodes différentes au choix
        """
        if not self.tau_update and self.episode%self.target_update==0:  #avec update périodique
            self.targetnet.set_weights(self.policynet.get_weights())
        else:                                                           # avec update continue et progressive
            q_model_theta = self.policynet.get_weights()
            target_model_theta = self.targetnet.get_weights()
            i = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                target_model_theta[i] = target_weight
                i += 1
            self.targetnet.set_weights(target_model_theta)

    def fitting(self):
        """
        fonction servant à faire converger le RN 
        on prend un batch de la mémoire, calcul une différence avec les résultats de Target, policy et le résultat trouvé
        puis fit le RN grace à ces données
        """
        #traitement des données
        experiences = self.samplememory(self.batchsize_fit)
        states, actions, rewards , next_states, dones =  utilities.extract_tensors(list(zip(*experiences))[1])
        #prédiction pour affiner les RN 
        target = self.policynet.predict([states[:][0],states[:][1]],verbose=0)
        target_next = self.policynet.predict([next_states[:][0],next_states[:][1]],verbose=0)
        target_val = self.targetnet.predict([next_states[:][0],next_states[:][1]],verbose=0)
        #calcul des erreurs
        erreur = np.zeros(self.batchsize_fit)
        for i in range(self.batchsize_fit):
            val = target[i][actions[i]]
            if dones[i]: 
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = self.gamma * target_val[i][np.argmax(target_next[i])] + rewards[i]
            erreur[i] = np.abs(val - target[i][actions[i]])
        #actualisation de la mémoire
        for i in range(self.batchsize_fit):
            self.updatememory(experiences[i][0], erreur[i])
        #fit
        self.policynet.fit([states[:][0],states[:][1]], target, batch_size=self.batch_size,sample_weight=None,verbose=0, epochs=self.epochs)
    
    def get_explo_rate(self):
        #simple fonction pour connaitre le taux d'exploration à un pas donné
        return self.end + np.exp(-1. * self.decay * self.current_step) * (self.start - self.end)

    def addmemory(self, experience):
        """
        on ajoute à la mémoire des données selon un format définit, en ajoutant une erreur
        pour pondérer l'arbre binaire de stockage
        """
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
        """
        on prend n éléments de la mémoire selon une loi de distribution donnée par l'erreur
        """
        batch = []
        moyenne = self.tree.tree[0] / n
        for i in range(n):
            batch.append(self.tree.get(np.random.uniform(moyenne * i, moyenne * (i + 1))))
        return batch

    def updatememory(self, idx, error):
        #on met à jour l'erreur de l'arbre de mémoire
        self.tree.update(idx, np.power(error + self.e, self.a))

    def select_action(self,state):
        """
        pour un état donné, on prend la décision de l'action à prendre 
        """
        self.current_step +=1
        if self.episode >= self.supervised :                    #inutile ici, seulement présent si on compte faire une partie purement supervisée
            if self.get_explo_rate() > np.random.random():      #algorithme en exploration explo_rate % du temps, sinon en exploitation
                return np.random.randint(0,self.actionspace)
            else:
                return tf.squeeze(tf.argmax(tf.squeeze(self.policynet(state)))).numpy()  
                #on prédit l'action/la puissance à faire/donner puis on prend l'indice de la valeur maximale comme étant le % de puissance à envoyer 
        else: return(self.supervise())

    def load(self,name,dir):
        """
        simple fonction utilitaire pour charger facilement un RN 
        """
        print(f"------------CHARGEMENT MODELE {name}------------")
        try:
            if name =="Target":self.targetnet = tf.keras.models.load_model(dir)
            else : self.policynet = tf.keras.models.load_model(dir)
            print(f"--------------MODELE CHARGE {name}--------------")
        except Exception as ex: print("échec du chargement : ",ex)

    def save(self,name,nb):
        """
        fonction pour sauvegarder les RN 
        """
        print(f"----------ENREGISTREMENT MODELE {name}----------")
        try:
            saveplace = os.path.join(self.dir, f"{name}{nb}.keras") 
            if name =="Target": self.targetnet.save(saveplace) 
            else : self.policynet.save(saveplace)

            print(f"------------MODELE ENREGISTRE {name}------------")
        except Exception as ex: print("échec de l'enregistrement : ",ex)
    
    def reset(self):
        # on reset tout l'environnement et la mémoire des RN 
        self.env.reset()
        self.current_step=0
        self.policynet.reset_states()

    def save_param(self):
        """
        on enregistre les données importantes dans une DB
        #fonction pas finie
        """
        try:
            conn = sqlite3.connect("TIPE/database/rewards.db")
            cur = conn.cursor()
            cur.execute(f'''CREATE TABLE reward (
            "id"	INTEGER,
            "x"	INTEGER,
            );''')
            conn.commit()
            for w in range(len(self.rewards)):
                newdata = (cur.lastrowid,self.rewards[w][0])
                cur.execute(f"INSERT INTO reward VALUES (?,?)",newdata)
                conn.commit()
        except Exception as e:
            print("[!ERREUR!]",e)
            conn.rollback()
        finally:
            conn.close()

    def supervise(self):
        pass

    def training(self):
        """
        boucle principale d'apprentissage
        """
        for episode in range(self.episodes):
            self.episode+=1
            self.reset()
            state = self.env.get_state()
            
            while not self.env.done:
                action = self.select_action(state)
                reward = self.env.take_action(action)
                next_state = self.env.get_state()
                self.addmemory(utilities.Experience(state,action,reward,next_state,self.env.done))
                state = next_state
                self.fitting()
            
            self.rewards[episode]=reward
            self.tautransfer()
            
            print(f"episode : {episode} || modele numero : {episode//self.checkpoint+1} || dernière récompense : {reward}")

            if episode % self.checkpoint ==0 and self.saving: self.save("Policy",episode//self.checkpoint+1)    #on enregistre périodiquement les modèles par sécurité  
        self.save_param()
        if self.saving: self.save("Policy",self.episodes//self.checkpoint+2) 
        
        
envs=environnement.ENV()
kevin = Agent(envs)
kevin.training()