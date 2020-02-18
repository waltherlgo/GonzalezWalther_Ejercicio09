import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

data = pd.read_csv('USArrests.csv')
Ciudades=np.array(data['Unnamed: 0'])
Names=["Murder","Assault","UrbanPop","Rape"]
Mat=np.array(data[Names])
for i in Names:
    globals()[i]=np.array(data[i])
    globals()[i]=(globals()[i]-np.mean(globals()[i]))/np.std(globals()[i])
Mat2=np.zeros((50,4))
for i in range(4):
    Mat2[:,i]=globals()[Names[i]]
S=np.zeros(4)
for i in Mat2:
    S=S+np.outer(i,i)
S=1/(Mat.shape[0]) *S
w,v=np.linalg.eig(S)
liste=np.flipud(np.argsort(w))
v1=v[:,liste[0]]
v2=-v[:,liste[1]]
pc=np.zeros(50)
sc=np.zeros(50)
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)
ax2.xaxis.set_label_position('top') 
ax2.yaxis.set_label_position('right') 
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.scatter([-1,1],[-1,1],color='white')
for i in range(50):
    pc[i],sc[i]=[v1@Mat2[i,:],v2@Mat2[i,:]]
    ax.scatter(pc[i],sc[i],color='black')
    ax.text(pc[i]+.04,sc[i]+.04,Ciudades[i])
for i in range(4):
    ax2.arrow(0,0,v1[i],v2[i],head_width=0.04,color='r',width=0.003)
    ax2.text(v1[i]+.05,v2[i]+.05,Names[i],color='r',fontsize=20)