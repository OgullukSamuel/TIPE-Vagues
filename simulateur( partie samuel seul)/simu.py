import numpy as np
import pygame as pg

class Environnement():
    def __init__(self):
        self.t=0
        self.dt=0.1                                 #en secondes
        self.reset()
        self.delta_angle=15                          #de cb de degré le bateau peu changer son orientation à chaque pas
        self.puissance_moteur=3                     #en Watt
        self.mass=0.140                                #en kg
        self.parcours=np.array([[0,0],[10,0],[12,2]])
        self.len_parcours=self.parcours.shape[0]
        self.marge=4                                #marge qu'on laisse pour atteindre le point B (au carré) en metres
        self.action_space=2                         #action sous forme :(puissance,dtheta)                         
        self.observation_space=(255,255,2)
        self.angle_wave=0                           #en degrés
        self.amplitude_wave=2                       #en metres
        self.omega_wave=2                           #en secondes^-1
        self.vitesse_wave=2                         #en metres/s
        self.nb_sample=3
        
    
    def get_frottement_force(self):
        L=0.1                           #longueur caractéristique ( largeur, longueur ou diagonale)
        visco_eau=9.3*1e-5
        Cx=1                            #coef de trainée
        S=5                             #surface normale aux frottements ! m^2
        sped=np.linalg.norm(self.vitesse)
        reynold=L*sped/visco_eau
        if reynold>1000:
            force=0.5*1*(sped**2)*Cx*S
        elif reynold < 20:
            force = Cx*sped
        else:
            force=Cx*sped**1.4
        
        return([force[0]*np.cos(self.angle),force[1]*np.sin(self.angle)])

    def get_wave_force(self):
        force = self.amplitude_wave*np.cos(np.add((self.omega_wave/self.vitesse_wave)*self.pos,self.omega_wave*self.t))
        force=[force[0]*np.cos(np.radians(self.angle_wave)),force[1]*np.sin(np.radians(self.angle_wave))]
        return(force)

    def get_moteur_force(self,action):
        return([action[0]*self.puissance_moteur*np.cos(self.angle),action[0]*self.puissance_moteur*np.sin(self.angle)])

    def render(self):
        pass

    def reset(self):
        self.vitesse=np.zeros(2)
        self.acceleration=np.zeros(2)
        self.pos=np.zeros(2)
        self.angle=0
        self.t=0
        self.pts=0
        

    def get_state(self):
        dists=np.linalg.norm([self.pos]*self.nb_sample-self.parcours[self.pts:self.pts+self.nb_sample-1])
        angles=np.arctan2(np.substract([self.pos]*self.nb_sample,self.parcours[self.pts:self.pts+self.nb_sample-1]))
        infos=[dists,angles]
        return infos

    def step(self,action):
        self.angle+=action[1]*self.delta_angle
        frottement=self.get_frottement_force()
        wave=self.get_wave_force()
        moteur=self.get_moteur_force(action)
        self.t+=self.dt
        
        self.acceleration=np.add(moteur,np.add(frottement,wave))/self.mass
        self.vitesse=np.add(self.vitesse,self.acceleration*self.dt)
        self.pos=np.add(self.pos,self.vitesse*self.dt)

        if np.linalg.norm(self.pos-self.parcours[self.pts])<=self.marge:self.pts+=1
        
        done= True if self.pts>=self.len_parcours else False
        
        reward=-1+0.1*self.pts/self.len_parcours if not done else 10
        
        state= self.get_state()
        return(state,reward,done)