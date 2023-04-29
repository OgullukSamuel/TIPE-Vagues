import numpy as np
import imageio.v3 as iio
from collections import namedtuple
import tensorflow as tf

def getwebcam(nb_img,enregistrer):
    if nb_img<0:return None
    frames=[]
    for frame_count, frame in enumerate(iio.imiter("<video0>")):
        if enregistrer :iio.imwrite(f"frame1_{frame_count}.jpg", frame)
        frames.append(frame)
        if frame_count > nb_img-2:
            break
    return(np.array(frames))

def get_grad(imglist):
    imgs=[]
    for img in imglist:
        #img1 = np.squeeze(np.hsplit(img,img.shape[1]))
        img2=np.gradient(img)
        imgs.append(np.add(img2[0],img2[1],img2[2]))
    imgs=np.clip(imgs,0,255)
    print(imgs.shape)
    return(np.array(imgs))

Experience = namedtuple("Experience",("state","action","reward","next_state","done"))

def extract_tensors(experiences):
    """
    J'extrais 5 tenseurs à partir d'un named tuples experiences 
    """
    batch = Experience(*zip(*experiences))
    t1 = tf.concat([batch.state],-1)
    t2 = np.concatenate([batch.action],-1)
    t3 = np.concatenate([batch.reward],-1)
    t4 = tf.concat([batch.next_state],-1)
    t5 = np.concatenate([batch.done],-1)
    return(t1,t2,t3,t4,t5)

class SumTree:
    write = 0
    """
    crée un arbre binaire (bibliothèque venant de Github)
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1)//2
        self.tree[parent] += change
        if parent != 0: self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree): return idx
        if s <= self.tree[left]: return self._retrieve(left, s)
        else: return self._retrieve(right, s-self.tree[left])

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity: self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])