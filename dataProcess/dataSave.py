from data import *
import pickle

fx = open("data/x", "wb")
fy = open("data/y", "wb")
fei = open("data/edge_index", "wb")
fea = open("data/edge_attr", "wb")
pickle.dump(x, fx)
pickle.dump(y, fy)
pickle.dump(edge_index, fei)
pickle.dump(edge_attr, fea)
fx.close()
fy.close()
fei.close()
fea.close()
