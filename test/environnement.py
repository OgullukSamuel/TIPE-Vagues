import numpy as np
import imageio.v3 as iio
import parametres
import time
class ENV():
    def __init__(self):
        self.batches=parametres.para.inputshape[0]
        self.energy=parametres.para.energy
        self.time_lim=parametres.para.tps_limite
        self.reset()
    
    def getwebcam(self,enregistrer):
        """
        acquiert des images de la webcam
        (par défaut laisser à 1) 
        """
        if self.batches<0:return None
        i=0
        frames=np.zeros(parametres.para.inputshape)
        for frame_count, frame in enumerate(iio.imiter("<video0>")):
            if enregistrer :iio.imwrite(f"frame1_{frame_count}.jpg", frame)
            frames[i]=frame
            i+=1
            if frame_count > self.batches-2: break
        self.state=frames

    def get_grad(self):
        """
        renvoie le gradient de l'image (utile pour détecter les bords)
        """
        imgs=np.zeros(self.state.shape)
        for i in range(self.state.shape[0]):
            img2=np.gradient(self.state[i])
            imgs[i]=np.add(img2[0],img2[1],img2[2])
        return(np.array(imgs))

    def get_context_map(self):
        """
        fonction pour avoir une carte de contexte (actuellement inutile, 
        mais utile si le format des données ne permet pas une convergence)
        cf travaux de chez google
        """
        self.getwebcam(False)
        return self.state  #temporaire

    def get_state(self):
        """
        fonction renvoyant l'état de l'environnement , traité et prête à aller dans l'agent
        (l'ordre du code est important)
        """
        context=self.get_context_map()
        grad = self.get_grad()
        

        return [grad,context]

    def take_action(self,action):
        """
        fonction prenant un % de la puissance à envoyer, envoie cette puissance dans le moteur puis renvoie une récompense
        """
        if self.energy_depensee > self.energy:
            pass  
            #on envoie aucune énergie au moteur
        else:
            pass
            # on envoie action% de la puissance du moteur
        
        reward = 0 
        #mettre systeme de récompense
        if time.time()> self.time+self.time_lim: self.done = True
        #condition pour vérifier si le temps est done
        return  reward 
    
    def reset(self):
        self.energy_depensee=0
        self.done=False
        self.time=time.time()
env=ENV()