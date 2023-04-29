import numpy as np
import imageio.v3 as iio

class ENV():
    def __init__(self):
        pass
    
    def getwebcam(nb_img,enregistrer):
        if nb_img<0:return None
        frames=[]
        for frame_count, frame in enumerate(iio.imiter("<video0>")):
            if enregistrer :iio.imwrite(f"frame1_{frame_count}.jpg", frame)
            frames.append(frame)
            if frame_count > nb_img-2: break
        return(np.array(frames))

    def get_grad(imglist):
        imgs=np.zeros(imglist.shape)
        for i in range(imglist.shape[0]):
            img2=np.gradient(imglist[i])
            imgs[i]=np.add(img2[0],img2[1],img2[2])
        return(np.array(imgs))