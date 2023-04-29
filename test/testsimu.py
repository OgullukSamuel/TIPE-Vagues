import numpy as np

class parametre():

    N = 1000   # nb de molécules
    m= 0.001   #masse d'une particule
    g=9.81
    size= 0.001
    rho=0.8    #masse volumique de l'agent
    grid_step = 0.1 #
    grid_size = int(1/grid_step)  
    v0 = [0,0]
    ship_dim = 0.2
    deter=True
    a0=[0,0]


        

class Environment():
    def __init__(self):
        self.param=parametre()
        self.N=self.param.N
        self.grid_size=self.param.grid_size
        self.billes_pos=np.zeros((self.N,2))
        self.billes=[]
        self.grid_sum = np.zeros((self.grid_size,self.grid_size))
        self.determinisme=self.param.deter
    
    def distribution_init(k,N):
        pos_bille_k= [(5*k)//N,k%(5*N)]
        return(pos_bille_k)

    def build_env(self):
        for i in range(self.N):
            pos=self.distribution_init(i,self.N)
            bille=self.billes(pos)
            self.billes.append(bille)
            self.billes_pos[i]=bille.pos
            self.grid_sum[int(pos[0]),int(pos[1])]+=1
        



        if self.determinisme:
            self.p0=(self.billes_pos,self.grid)

        
    def update_Collision(self):
        A=np.where(x>1,self.grid_sum)
        print(A)



    def reset(self):
        if self.determinisme:
            self.billes_pos,self.grid = self.p0
        else:
            for i in range(self.N):
                pos=self.distribution(i,self.N)
                bille=self.billes(pos)
                self.billes.append(bille)
                self.billes_pos[i]=bille.pos
                self.grid[int(pos[0]),int(pos[1])]+=1

    def step(self,action):
        pass

    def get_reward(self):
        pass

    def show_env(self):
        pass

    def save_env(self,path):
        pass
    
    def get_state(self):
        pass

class billes():
    def __init__(self,pos):
        param=parametre()
        self.v = param.v0
        self.pos=pos
        self.size=param.size
        self.a = param.a0
    
    def step(self):
        newpos=0
        return(newpos)
    


