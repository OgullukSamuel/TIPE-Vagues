import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class Environnement():
    def __init__(self,c_init=()):                      #condition initiales sous le format :([v0x,v0y],[a0x,a0y],pos0,angle0,t0,pts0)
        """
        on initialise l'environnement
        """
        self.dt=0.02                                        #en secondes
        
        self.parcours=np.array([[5,0]])     # les points que le bateau doit passer
        self.len_parcours=self.parcours.shape[0]            # nb de points du parcours
        self.marge=2                                        #marge qu'on laisse pour atteindre le point B (au carré) en metres
        self.action_space=6                                 #action sous forme :(puissance,dtheta)           
        self.nb_sample=3                            #nb de points qu'on prend pour envoyer au rn ( pour prédire sur plusieurs checkpoints à l'avance)   doit etre inf à len_parcours
        self.observation_space=(2,self.nb_sample,2)
        self.inputshape1=(self.nb_sample,2)
        self.inputshape2=10              
        

        self.delta_angle=0.1                               #de cb de degré le bateau peu changer son orientation à chaque pas
        self.puissance_moteur=30                            #en Watt   , puissance moteur du bateau
        self.mass=3                                         #en kg, masse du bateau
        
        self.angle_wave=15                                   #en degrés
        self.amplitude_wave=0.5                               #en metres ( à multiplier par 2)  
        self.periode_wave=12                                #en secondes
        self.omega_wave=2*np.pi/self.periode_wave           #en secondes^-1
        self.profondeur=1                                   #en metre , profondeur de l'eau au point étudié
        self.vitesse_wave=np.sqrt(9.81*self.profondeur)     #en metres/s  https://www.surf-report.com/news/pedagogie-meteo/periode-swell-houle-vague-vagues-houles-1128196795.html        
        self.k= (self.omega_wave/(np.sqrt(9.81*self.profondeur)))*\
            np.array([np.cos(np.radians(self.angle_wave)),np.sin(np.radians(self.angle_wave))]) # vecteur d'onde de la vague étudiée
        self.A_vit=self.amplitude_wave*self.omega_wave/(2*np.sinh(np.linalg.norm(self.k) * self.profondeur))    #L'amplitude" de la vitesse
        L=0.40                                              #longueur caractéristique largeur du drone
        visco_eau=1e-3                                      #source : https://wiki.anton-paar.com/fr-fr/eau/
        rho=1000                                            # masse volumique (kg/m3)
        S=0.05                                              #surface normale aux frottements ! m^2
        self.coef=L*rho/visco_eau                           #coef relatif aux frottements de l'eau ( servant à calculer le nombre de Reynolds)
        self.coef2=0.5*rho*S                                #de même pour calculer les frottements visqueux
        self.vitesse_seuil= 0.3                             #en metre/s , vitesse en dessous de laquelle le moteur impose une force constance
        self.puissance_moteur_seuil= 30                     #puissance moteur au démarrage
        self.energy_max=50000
        self.t_max=5                                        #temps maximal
        self.reset(c_init)


    def get_angle(self,x):
        """
        prend une liste de la forme [[x1,y1],[x2,y2],...] et retourne la liste des arguments des vecteurs 
        """
        return(np.degrees(np.arctan2(np.array(x)[:,1],np.array(x)[:,0])))
    
    def get_frottement_force(self):                      
        """
        donne la force des frottements en fonction de la vitesse et d'autres parametres prédéfinis
        on regarde 3 régimes, laminaire, turbulent et intermédiaire en fonction du nombre de Reynolds de
        la situation étudiée, puis on calcule les frottements associés

        pour calculer le Cx, on utilise un polynome du second degré 
        """
        sped=np.linalg.norm(self.vitesse)
        if sped==0:
            return([0,0])
        reynold=self.coef*sped
        deviation = self.get_angle([self.vitesse])[0]-self.angle
        Cx=0.0116+deviation*(-1.34e-3)+(deviation**2)*(2.66e-04)    
        if reynold>1000:
            force=self.coef2*(sped**2) *Cx
        elif reynold < 30:
            force = self.coef2*sped*Cx
        else:
            force=Cx*(sped**1.4) *self.coef2
        return([-force*np.cos(np.radians(deviation+self.angle)),-force*np.sin(np.radians(deviation+self.angle))])

    def get_wave_force(self):
        """
        donne la force de la vague sur le bateau
        on calcule la vitesse des particules d'eau de la vague, puis leur force
        """
        etat_vague = np.add(np.dot(self.pos,self.k),self.omega_wave*self.t)     # on calcule ici l'état de la vague (phase en un point + wt)
        vague = self.amplitude_wave*np.sin(etat_vague)
        force = np.power(np.dot(self.A_vit*np.cosh(np.linalg.norm(self.k)*(self.profondeur+vague)),self.k),2)/2
        return(force,vague)
    

    def get_moteur_force(self,action):
        """
        donne la force développée par le moteur en fonction de la vitesse
        l'action est ici donnée entre 0 et 1, pour une puissance constante ayant pour valeur soit puissance moteur
        soit puissance moteur seuil, en fonction de la vitesse ( selon P=Fv <=> F=P/v)
        """
        action=action/2+0.5
        if np.linalg.norm(self.vitesse)<=self.vitesse_seuil:
            return(np.array([np.cos(np.radians(self.angle)),np.sin(np.radians(self.angle))])*action*self.puissance_moteur_seuil)
        else :
            return(np.array([np.cos(np.radians(self.angle)),np.sin(np.radians(self.angle))]) *action*self.puissance_moteur/np.linalg.norm(self.vitesse))

    def reset(self,c_init=()):
        """
        on réinitialise les parametres des épisodes
        """
        if c_init==():
            self.vitesse=np.zeros(2) 
            self.acceleration=np.zeros(2) 
            self.pos=np.zeros(2) 
            self.angle=0
            self.t=0
            self.pts=0
        else:
            self.vitesse,self.acceleration,self.pos,self.angle,self.t,self.pts = c_init
        return(self.get_state())



    def get_state(self):
        """
        on obtient un état de notre environnement (tous les parametres importants qui seront ensuite traités dans les réseaux neuronal)
        """
        dists=np.zeros((self.nb_sample))
        if self.pts+self.nb_sample>self.len_parcours:
            v=self.parcours[self.pts:]-self.pos  
            for k in range(self.len_parcours-self.pts): 
                dists[k]=np.linalg.norm(v[k]) 
            angles=np.hstack((self.get_angle(v),np.zeros((self.nb_sample-self.len_parcours+self.pts))))
        else:
            v=self.parcours[self.pts:self.pts+self.nb_sample]-self.pos
            for k in range(self.nb_sample): 
                dists[k]=np.linalg.norm(v[k]) 
            angles=self.get_angle(v)
        info=np.array([self.t,self.angle,self.vitesse[0],self.vitesse[1],self.angle_wave,self.omega_wave,self.amplitude_wave,self.vitesse_wave,self.puissance_moteur,self.mass])

        states=[np.dstack((dists,angles)).squeeze(),info]
        state=[np.array([states[0]]),np.array([states[1]])]
        temp =np.concatenate((state[0].flatten(),state[1].flatten()),axis=0)
        return temp

    def step(self,action):
        """
        action sous la forme de one hot encoding :
        0 -> aucune puissance ni angle
        1-> on tourne à gauche, puissance nulle
        2-> on tourne à gauche, pleine puissance
        3-> on tourne pas, pleine puissance
        4-> on tourne à droite, puissance nulle
        5-> on tourne à droite, pleine puissance
        """
        self.angle+=(-action[1]-action[2]+action[4]+action[5])*self.delta_angle
        self.angle%=360
        frottement=[0,0]#self.get_frottement_force()
        wave=0#self.get_wave_force()[0]
        moteur=self.get_moteur_force(action[2]+action[3]+action[5])
        self.t+=self.dt
        self.acceleration=np.add(moteur,np.add(frottement,wave))/self.mass
        self.vitesse=np.add(self.vitesse,self.acceleration*self.dt)
        self.pos=np.add(self.pos,self.vitesse*self.dt)
        dist =np.linalg.norm(self.pos-self.parcours[self.pts])
        
        done= 1 if ((self.pts==self.len_parcours and dist<=self.marge) or self.t>self.t_max) else 0
        
        if done==0:
            reward=self.pts/self.len_parcours - dist
        else: 
            reward=1000
        
        
        if dist<=self.marge:
            self.pts+=1
            reward=200

        state= self.get_state()
        return(state,reward,done)

    def plot_state(self,t=0):
        x = y = np.arange(-50.0, 50.0,1)

        X, Y = np.meshgrid(x, y)
        xs = np.array(np.meshgrid(x,y))
        points = np.transpose(xs.reshape(2,-1))
        V, Z = self.get_wave_force_for_plot(points,(len(x),len(y)),t)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(45,60)


        ax.plot_surface(X, Y, Z, facecolors=cm.Oranges(V))
        plt.show()

    def get_wave_force_for_plot(self,pos,shape,t=0):
        """
        donne la force de la vague sur le bateau
        on calcule la vitesse des particules d'eau de la vague, puis leur force
        """
        n=len(pos)
        vague=np.zeros(n)
        force=np.zeros(n)
        vitesse=np.zeros(n)
        for i in range(n):
            etat_vague = np.add(np.dot(pos[i],self.k),self.omega_wave*t)
            vague[i]= self.amplitude_wave*np.sin(etat_vague)
            vitesse[i]=np.linalg.norm(self.A_vit*np.sin(np.dot(self.k,pos[i])+self.omega_wave*t))
            force[i]= np.linalg.norm(np.power(np.dot(self.A_vit*np.cosh(np.linalg.norm(self.k)*(self.profondeur+vague[-1])),self.k),2)/2)

        return(np.reshape(vitesse,shape),np.reshape(vague,shape))

a=Environnement()
a.plot_state()