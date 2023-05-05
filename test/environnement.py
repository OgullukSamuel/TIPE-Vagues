import numpy as np
import imageio.v3 as iio
import parametres

class ENV():
    def __init__(self):
        self.batches=parametres.para.inputshape[0]
        self.state=self.getwebcam(False)
        self.done=False
        self.loutre =0
    
    def getwebcam(self,enregistrer):
        if self.batches<0:return None
        i=0
        frames=np.zeros(parametres.para.inputshape)
        for frame_count, frame in enumerate(iio.imiter("<video0>")):
            if enregistrer :iio.imwrite(f"frame1_{frame_count}.jpg", frame)
            frames[i]=frame
            i+=1
            if frame_count > self.batches-2: break
        return(frames)

    def get_grad(self):
        imgs=np.zeros(self.state.shape)
        for i in range(self.state.shape[0]):
            img2=np.gradient(self.state[i])
            imgs[i]=np.add(img2[0],img2[1],img2[2])
        return(np.array(imgs))

    def get_context_map(self):
        return self.getwebcam(False)  #temporaire

    def get_state(self):
        #condition pour changer env.done
        grad = self.get_grad()
        context=self.get_context_map()

        return [grad,context]

    def take_action(self,action):
        reward = 0
        
        self.loutre+=1
        if self.loutre> 10 : self.done = True               # agit sur le moteur en fonction de l'action
        return reward,self.done 
    
    def reset(self):
        pass
env=ENV()