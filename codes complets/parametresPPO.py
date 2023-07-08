import tensorflow as tf

class para:
    """
    Classe de tous les paramètres modifiables
    """   
    episodes = 50                      

    dir="model"				            # fichier où j'enregistre le modèle
    save = True	                        #enregistrement des RN à chaque checkpoint 


        #hyperparametres
    LearningRate = 1e-3
    actor_opti=tf.keras.optimizers.Adam(learning_rate=LearningRate)
    critic_opti=tf.keras.optimizers.Adam(learning_rate=LearningRate)
    actor_loss=tf.keras.losses.CategoricalCrossentropy()          
    critic_loss=tf.keras.losses.MeanSquaredError()
    actor_metrique=[tf.keras.metrics.CategoricalCrossentropy()] 
    critic_metrique=[tf.keras.metrics.MeanSquaredError()]                                  	
    initializer= tf.keras.initializers.RandomNormal(stddev=0.25,mean=0.5)						                        

        # gestion du target+Memoire
    batch_size = 10                     	    # taille du batch de Replay memory sur lequel j'entraine le modèle à chaque tour

    supervised=0                            #nb de tours en supervisés

    gamma = 0.999                       	# taux de diminution de la récompense
    lmbda=0.95
    clip=0.2

    inputshape = (1,4)             #taille de l'input dans les RN , (laisser le 1)
    observation_shape=(1,1,1)
    nbimage=3                                  #nombre d'images mise dans l'état
    action_space = 100                       #nombre d'actions possibles ( marge de manoeuvre sur les %)
    epochs = 15                              # nombre d'epochs lors de l'entrainement
    checkpoint=20                           #episodes avant un checkpoint

    #variables d'environnement
    energy = 200                            #nombre énergie disponible ( delta E)
    tps_limite=10                           #temps d'un episode ( en secondes)
    boatcolor= [0,255,0]                    #BGR
    watercolor=[255,0,0]
    color_accuracy=20