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

        # gestion du target+Memoire
    batch_size = 1                     	    # taille du batch de Replay memory sur lequel j'entraine le modèle à chaque tour

    supervised=0                            #nb de tours en supervisés

    gamma = 0.999                       	# taux de diminution de la récompense
    lmbda=0.95
    clip=0.2

    inputshape = (1,480,640,3)             #taille de l'input dans les RN , (laisser le 1)
    observation_space=(1,1,1)
    nbimages=1                                  #nombre d'images mise dans l'état
    action_space = 40                       #nombre d'actions possibles ( marge de manoeuvre sur les %)
    epochs = 15                              # nombre d'epochs lors de l'entrainement
    checkpoint=20                           #episodes avant un checkpoint

    #variables d'environnement
    energy = 200                            #nombre énergie disponible ( delta E)
    amplitude=0.13                          #amplitude de la vague ( en pratique c'est 1/pulsation) ( entre 0.2 et 0.5),
    episode_duration=10