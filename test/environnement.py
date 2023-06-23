import numpy as np
import cv2 as cv
import parametres
import time
import pyfirmata

class ENV():
    def __init__(self):
        self.inputshape=parametres.para.inputshape
        self.batch=self.inputshape[0]
        self.energy=parametres.para.energy
        self.time_lim=parametres.para.tps_limite
        self.vc = cv.VideoCapture(0,cv.CAP_MSMF)
        self.reset()
        hsv_bateau = cv.cvtColor(np.uint8([[parametres.para.boatcolor]]),cv.COLOR_BGR2HSV)
        self.lower_bateau = np.array([hsv_bateau[0][0][0]-parametres.para.color_accuracy,50,50])
        self.upper_bateau = np.array([hsv_bateau[0][0][0]+parametres.para.color_accuracy,255,255])
        hsv_eau = cv.cvtColor(np.uint8([[parametres.para.watercolor]]),cv.COLOR_BGR2HSV)
        self.lower_eau = np.array([hsv_eau[0][0][0]+parametres.para.color_accuracy,50,50])
        self.upper_eau = np.array([hsv_eau[0][0][0]+parametres.para.color_accuracy,255,255])
        self.cX,self.cY=0,0
        #self.board = pyfirmata.Arduino('COM6') # Windows
        #self.engine = self.board.digital[3]
        #self.engine.mode = pyfirmata.PWM
        _, self.frem = self.vc.read()

    def get_state(self):
        """
        retourne un état traité sous la forme : (état, contexte)
        au réseau de neurone
        """
        
        context=np.zeros((self.batch,3,2))
        frames=np.zeros((self.inputshape))
        states=np.zeros((self.inputshape))
        for i in range(self.batch):
            _, frame = self.vc.read()
            frame=self.frem
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            frames[i]=frame
            mask = cv.inRange(hsv, self.lower_bateau, self.upper_bateau)
            mask_eau = cv.inRange(hsv, self.lower_eau, self.upper_eau)
            bateau=cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((4,4),np.uint8))
            eau=cv.morphologyEx(mask_eau, cv.MORPH_OPEN, np.ones((4,4),np.uint8))
            grad=cv.Canny(frame,100,200)
            states[i]=np.stack((bateau,eau,grad),axis=-1)
            M = cv.moments(bateau)
            if M["m00"] != 0:
                context[i][0]=[self.cX-M["m10"] / M["m00"], self.cY-M["m01"] / M["m00"]]
                self.cX,self.cY=M["m10"] / M["m00"], M["m01"] / M["m00"]
                context[i][2]=[self.cX,self.cY]
        return(frames,states,context)


    """
    def take_action(self,action):
        #
        #fonction prenant un % de la puissance à envoyer, envoie cette puissance dans le moteur puis renvoie une récompense
        # 
        if self.energy_depensee > self.energy:
            self.engine.write(0)
            #on envoie aucune énergie au moteur
        else:
            self.engine.write(action*0.9+0.1)
            # on envoie action% de la puissance du moteur
        
        reward = 0
        #mettre systeme de récompense
        if time.time()> self.time+self.time_lim: self.done = True
        #condition pour vérifier si le temps est done
        
        return reward 
    """
    def reset(self):
        self.energy_depensee=0
        self.done=False
        self.time=time.time()
        

        #self.engine.write(0)
env=ENV()
t=time.perf_counter()
for _ in range(100):
    env.get_state()
print(time.perf_counter()-t)
#env.vc.release()