import numpy as np
import cv2 as cv
import parametresPPO as parametres
import time
import pyfirmata

class ENV():
    def __init__(self):
        para=parametres.para()
        self.inputshape=para.inputshape
        self.batch=para.nbimages
        self.energy=para.energy
        self.episode_duration=para.episode_duration
        self.vc = cv.VideoCapture(0,cv.CAP_MSMF)
        self.action_space=para.action_space
        self.observation_space=para.observation_space

        self.lower_eau = np.array([85,50,50])
        self.upper_eau = np.array([115,255,255])
        self.cX,self.cY=0,0
        print("loutre")
        self.board = pyfirmata.Arduino('COM7') # Windows
        print("lour")
        self.bav_arriere= self.board.digital[10] #arriere
        self.bav_avant= self.board.digital[11] #avant
        self.engine = self.board.digital[3]
        self.engine.mode = pyfirmata.PWM
        self.reset()

    def get_state(self):
        """
        retourne un état traité sous la forme : (état, contexte)
        au réseau de neurone
        """
        context=np.zeros((self.batch,3,2))
        frames=np.zeros((self.inputshape),dtype=int)
        states=np.zeros((self.inputshape))
        for i in range(self.batch):
            _, frame = self.vc.read()
            if frame is None: print("frame is none")
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            frames[i]=frame
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, thresh = cv.threshold(gray, 35, 255, cv.THRESH_BINARY_INV)
            mask_eau = cv.inRange(hsv, self.lower_eau, self.upper_eau)
            bateau=cv.morphologyEx(thresh, cv.MORPH_OPEN, np.ones((4,4),np.uint8))
            eau=cv.morphologyEx(mask_eau, cv.MORPH_OPEN, np.ones((4,4),np.uint8))
            grad=cv.Canny(frame,100,200)
            states[i]=np.stack((bateau,eau,grad),axis=-1)
            M = cv.moments(bateau)
            if M["m00"] != 0:
                context[i][0]=[self.cX-M["m10"] / M["m00"], self.cY-M["m01"] / M["m00"]]
                self.cX,self.cY=M["m10"] / M["m00"], M["m01"] / M["m00"]
                context[i][2]=[self.cX,self.cY]
        return([frames,states,context])

    def move(self,sens,temps,timeless=False):
        """
        0-1=arriere
        1= avant
        temps = 1/tau
        """
        if sens == 1:
            self.bav_arriere.write(0)
            self.bav_avant.write(1)
            if not timeless:
                time.sleep(temps)
                self.bav_avant.write(0)
        else:
            self.bav_avant.write(0)
            self.bav_arriere.write(1)
            if not timeless:
                time.sleep(temps)
                self.bav_arriere.write(0)

    def create_wave(self,temps=10,amplitude=0.23):
        self.move(0,0.2)
        t=time.time()
        while time.time()<t+temps:
            self.move(0,amplitude)
            self.move(1,amplitude)
        self.move(1,0.2)
    
    def take_action(self,action):
        """"
            fonction prenant un % de la puissance à envoyer, envoie cette puissance dans le moteur puis renvoie (état,récompense,done)
        """
        state=self.get_state()
        deltat=time.time()-self.t
        self.t=time.time()
        self.energy_depensee+=(action*3.49+3.04)*deltat
        if self.energy_depensee > self.energy:
            self.engine.write(0)
            #on envoie aucune énergie au moteur
        else:
            self.engine.write(action+0.6)
            # on envoie action% de la puissance du moteur
        
        done = True if time.time()> self.time+self.episode_duration else False 
        #print(state[0])
        reward=abs(state[2][0][2][0]) if done else 0

        return (state,reward,done) 
    
    def reset(self):
        self.energy_depensee=0
        self.t=time.time()
        self.time=time.time()
        self.engine.write(0)
        state=self.get_state()
        return state
    
#env=ENV()
#cv.imshow("test",np.array(np.array(env.get_state()[1])[0]))
#            cv.imshow("testes",cv.bitwise_and(frame,frame, mask= mask_eau))
#k=cv.waitKey(0)
#env.create_wave(90)
#print(env.take_action(0.2)[0][2][0][2])