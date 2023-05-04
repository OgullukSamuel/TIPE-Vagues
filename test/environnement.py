import numpy as np
import imageio.v3 as iio
import parametres

class ENV():
    def __init__(self):
        self.batches=parametres.para.inputshape[0]
        self.state=self.getwebcam(False)
        self.done=False
    
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
        pass

    def get_state(self):
        return()

    def take_action(self,action):
        pass # agit sur le moteur en fonction de l'action

env=ENV()