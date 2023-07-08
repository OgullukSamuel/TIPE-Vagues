import tensorflow as tf

class para:
    """
    Classe de tous les paramètres modifiables
    """   
    episodes = 100                      

    dueling = True                      # True si mise en place de double dueling networks
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
    batch_size = 2                     	    # taille du batch de Replay memory sur lequel j'entraine le modèle à chaque tour
    batchsize_fit = 2                      	# taille du batch avec lequel j'utilise la fonction fit

    supervised=0                            #nb de tours en supervisés
    clip_param=0.2

    gamma = 0.999                       	# taux de diminution de la récompense
    lmbda=0.95
    clip=0.2

    inputshape = (1,4)             #taille de l'input dans les RN , (laisser le 1)
    action_space = 2                       #nombre d'actions possibles ( marge de manoeuvre sur les %)
    epochs = 15                              # nombre d'epochs lors de l'entrainement
    checkpoint=20                           #episodes avant un chackpoint