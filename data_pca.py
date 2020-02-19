import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

def funplot(file,Cols,VarName):
    data = pd.read_csv(file)
    data=data.fillna(0)
    #Lista de todas las columnas
    Columns=np.array(data.columns)
    Vars=Columns[Cols]
    Labels=np.array(data[Columns[VarName]])
    Mat=np.array(data[Vars])
    ND=Mat.shape[0]
    NV=Mat.shape[1]
    for i in Vars:
        globals()[i]=np.array(data[i])
        globals()[i]=(globals()[i]-np.mean(globals()[i]))/np.std(globals()[i])
    Mat2=np.zeros(Mat.shape)
    for i in range(NV):
        Mat2[:,i]=globals()[Vars[i]]
    S=np.zeros(NV)
    for i in Mat2:
        S=S+np.outer(i,i)
    S=1/(Mat.shape[0]) *S
    w,v=np.linalg.eig(S)
    liste=np.flipud(np.argsort(w))
    w=np.flipud(np.sort(w))
    vl=np.zeros((NV,NV))
    for i in range(NV):
        vl[i,:]=v[:,liste[i]]
    return vl,Mat2,ND,Labels,NV,Vars,w

def plotPC(PC1,PC2,vl,ND,name):
    pc=np.zeros(ND)
    sc=np.zeros(ND)
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111, label="1")
    plt.xlabel("Primera Componente Principal")
    plt.ylabel("Segunda componente Principal")
    ax2=fig.add_subplot(111, label="2", frame_on=False)
    ax2.xaxis.set_label_position('top') 
    ax2.yaxis.set_label_position('right') 
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.scatter([-1,1],[-1,1],color='white')
    for i in range(ND):
        pc[i],sc[i]=[vl[PC1,:]@Mat2[i,:],vl[PC2,:]@Mat2[i,:]]
        ax.scatter(pc[i],sc[i],color='black')
        ax.text(pc[i]+.04,sc[i]+.04,Labels[i])
    for i in range(NV):
        ax2.arrow(0,0,vl[PC1,i],vl[PC2,i],head_width=0.04,color='r',width=0.003)
        ax2.text(vl[PC1,i]+.05,vl[PC2,i]+.05,Vars[i],color='r',fontsize=20)
    plt.savefig(name)
    plt.show()
def plotVarE(w,name):
    plt.figure(figsize=(6,6))
    sw=np.zeros(len(w))
    for i in range(len(w)):
        sw[i]=np.sum(w[:i+1])/np.sum(w)
    plt.plot(range(1,len(w)+1),sw*100,"-o")
    plt.ylim(0,110)
    plt.xlabel("Numero de autovalores")
    plt.ylabel("Porcentaje de Varianza explicada")
    plt.grid()
    plt.savefig(name)
    plt.show()

vl,Mat2,ND,Labels,NV,Vars,w=funplot("USArrests.csv",[1,2,3,4],0)
plotPC(0,1,vl,ND,"arrestos.png")
plotVarE(w,"varianza_arrestos.png")
vl,Mat2,ND,Labels,NV,Vars,w=funplot("Cars93.csv",[7,8,12,13,14,15,17,19,21,22,25],2)
plotPC(0,1,vl,ND,"cars.png")
plotVarE(w,"varianza_cars.png")