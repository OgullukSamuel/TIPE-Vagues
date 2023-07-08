import numpy as np
from collections import namedtuple
import sqlite3
import pickle

Experience = namedtuple("Experience",("state","action","reward","next_state","done"))

def extract_tensors(experiences):
    """
    J'extrais 5 tenseurs à partir d'un named tuples experiences 
    """
    batch = Experience(*zip(*experiences))
    t1 = np.squeeze(np.concatenate([batch.state],-1))
    t2 = np.concatenate([batch.action],-1)
    t3 = np.concatenate([batch.reward],-1)
    t4 = np.squeeze(np.concatenate([[batch.next_state]],-1))
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
        return [idx, self.data[dataIdx]]
   
def save_blob(db, name, data, table_name="arrays"):
    pickled_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    db.execute(f"INSERT INTO {table_name} (name, data) VALUES (?, ?) ON CONFLICT DO UPDATE SET data=data",
        (name, pickled_data),)


def load_blob(db, name, table_name="arrays"):
    res = db.execute(f"SELECT data FROM {table_name} WHERE name = ?", (name,))
    row = res.fetchone()
    if not row:
        raise KeyError(name)
    return pickle.loads(row[0])


def create_blob_table(db, table_name="arrays"):
    db.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (name TEXT PRIMARY KEY NOT NULL, data BINARY NOT NULL)")