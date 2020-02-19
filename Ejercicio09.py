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
    
    
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
archivo_name='Cars93.csv'
data = pd.read_csv(archivo_name)
data=data.fillna(0)
#Lista de todas las columnas
Columns=np.array(data.columns)
#Seleccionamos la columna de la variable Y
VarY=27
#Descartamos las columnas innecesarias
ColI=[0,1,2,3,9,10,11,16,26]
Names=np.delete(Columns,np.append(VarY,ColI))
Ciudades=np.array(data[Columns[VarY]])
Mat=np.array(data[Names])
ND=Mat.shape[0]
NV=Mat.shape[1]
for i in Names:
    globals()[i]=np.array(data[i])
    globals()[i]=(globals()[i]-np.mean(globals()[i]))/np.std(globals()[i])
Mat2=np.zeros(Mat.shape)
for i in range(NV):
    Mat2[:,i]=globals()[Names[i]]
S=np.zeros(NV)
for i in Mat2:
    S=S+np.outer(i,i)
S=1/(Mat.shape[0]) *S
w,v=np.linalg.eig(S)
liste=np.flipud(np.argsort(w))
vl=np.zeros((NV,NV))
for i in range(NV):
    vl[i,:]=v[:,liste[i]]
pc=np.zeros(ND)
sc=np.zeros(ND)
def plotPC(PC1,PC2):
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)
    ax2.xaxis.set_label_position('top') 
    ax2.yaxis.set_label_position('right') 
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.scatter([-1,1],[-1,1],color='white')
    for i in range(ND):
        pc[i],sc[i]=[vl[PC1,:]@Mat2[i,:],vl[PC2,:]@Mat2[i,:]]
        ax.scatter(pc[i],sc[i],color='black')
        ax.text(pc[i]+.04,sc[i]+.04,Ciudades[i])
    for i in range(NV):
        ax2.arrow(0,0,vl[PC1,i],vl[PC2,i],head_width=0.04,color='r',width=0.003)
        ax2.text(vl[PC1,i]+.05,vl[PC2,i]+.05,Names[i],color='r',fontsize=20)
    plt.show()
plotPC(0,1)
plotPC(0,2)
plotPC(1,2)
