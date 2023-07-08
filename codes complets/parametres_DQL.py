import tensorflow as tf

class para:
    """
    Classe de tous les paramètres modifiables
    """   
    episodes = 100                      

    dueling = True                      # True si mise en place de double dueling networks
    tau_update=True			            # True si je veux soft update du réseau cible
    dir="model"				            # fichier où j'enregistre le modèle
    save = True	
    modeload = False			        
        # données de compilation
    LearningRate = 1e-3
    opti=tf.keras.optimizers.Adam(learning_rate=LearningRate)    	
    loss=tf.keras.losses.MeanSquaredError()                                          	
    metrique=[tf.keras.metrics.MeanSquaredError()]                                   	
    initializer= tf.keras.initializers.RandomNormal(stddev=0.25,mean=0.5)						                        

        # exploration ou exploitation
    eps_st = 1											# valeur d'exploration/ exploitation au départ
    eps_end = 0.01										# valeur d'exploration/ exploitation à la fin
    eps_decay = 0.001									# vitesse à laquelle l'algorithme passe d'explorateur à exploiteur

        # gestion du target+Memoire
    target_update = 16                  	# nombre de tour entre chaque actualisation du target net ( si soft update est désactivé )
    memory_size = 20000                  	# taille de la Replay memory
    batch_size = 20                     	    # taille du batch de Replay memory sur lequel j'entraine le modèle à chaque tour
    batchsize_fit = 20                      	# taille du batch avec lequel j'utilise la fonction fit
    tau=0.1					                # valeur de soft update
    PER_epsi = 0.01				            # valeurs servant à définir la perte pour choisir les données sur lesquelles entrainer l'algorithme
    PER_alpha = 0.6				            #

    supervised=0                            #nb de tours en supervisés

    gamma = 0.999                       	# taux de diminution de la récompense

    inputshape = (1,4)             #taille de l'input dans les RN , (laisser le 1)
    actionspace = 2                       #nombre d'actions possibles ( marge de manoeuvre sur les %)
    epochs = 10                              # nombre d'epochs lors de l'entrainement
    checkpoint=20                           #episodes avant un chackpoint

    energy = 200                            #nombre énergie disponible ( delta E)
    tps_limite=10                           #temps d'un episode ( en secondes)
    boatcolor= [0,255,0]                    #BGR
    watercolor=[255,0,0]
    color_accuracy=20