import tensorflow as tf

class para:
    """
    Classe de tous les paramètres modifiables
    """   
    episodes = 700                      

    dueling = True                      # True si mise en place de double dueling networks
    tau_update=True			            # True si je veux soft update du réseau cible
    dir="model"				            # fichier où j'enregistre le modèle
    save = 0		                    
    modeload = False			        

        # données de compilation
    LearningRate = 1e-3										                                    
    opti=tf.keras.optimizers.RMSprop(learning_rate=LearningRate, rho=0.95, epsilon=0.01)    	
    loss=tf.keras.losses.CategoricalCrossentropy()                                          	
    metrique=[tf.keras.metrics.CategoricalCrossentropy()]                                   	
    initializer= tf.keras.initializers.RandomNormal(stddev=0.25,mean=0.5)						                        

        # exploration ou exploitation
    eps_st = 1											# valeur d'exploration/ exploitation au départ
    eps_end = 0.01										# valeur d'exploration/ exploitation à la fin
    eps_decay = 0.001									# vitesse à laquelle l'algorithme passe d'explorateur à exploiteur

        # gestion du target+Memoire
    target_update = 16                  	# nombre de tour entre chaque actualisation du target net ( si soft update est désactivé )
    memory_size = 20000                  	# taille de la Replay memory
    batch_size = 2                     	# taille du batch de Replay memory sur lequel j'entraine le modèle à chaque tour
    batchsize = 32                      	# taille du batch avec lequel j'utilise la fonction fit
    tau=0.1					                # valeur de soft update
    PER_epsi = 0.01				            # valeurs servant à définir la perte pour choisir les données sur lesquelles entrainer l'algorithme
    PER_alpha = 0.6				            #

    supervised=0     #nb de tours en supervisés

    gamma = 0.999                       	# taux de diminution de la récompense

    inputshape = (1,720,1280,3)
    actionspace = 100
    epochs = 3

    target_update=20
    checkpoint=20
