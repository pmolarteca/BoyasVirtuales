 # -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:13:46 2018

@author: Unalmed
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import datetime
import scipy.stats
import xlsxwriter as xlsxwl # Crear archivos de Excel
import pandas as pd
import scipy.stats as scp
from windrose import WindroseAxes
from datetime import datetime 

path='./BV_34.txt'
Data = np.genfromtxt(path,delimiter='',dtype=str,skip_header=0)
    

Fechas = []
Parametros = np.zeros([len(Data), 4])
for i in range(len(Data)):
        Fechas.append(datetime.strptime(' '.join(Data[i,:4]), '%Y %m %d %H'))
        Parametros[i,:] = np.array(Data[i,4:]).astype(np.float)
Fechas = np.array(Fechas)

columns=['SWH', 'PERIODO', 'DIRECCION', 'SPREAD']
Waves = pd.DataFrame(Parametros, index=Fechas, columns=columns)

#plt.plot(Fechas,Waves[1])


#for i in range (29,35):
 #   path='./BV_'+str(i)+'.txt'
  #  Data = np.genfromtxt(path,delimiter='',dtype=str,skip_header=0)
    
   
##ciclo anual

WavesM=Waves.SWH.resample('M').mean()
#WavesD=Waves[0].resample('D').mean()

WM=np.array(WavesM)
WM=np.reshape(WM,(-1,12))
WMM=np.mean(WM,axis=0)
WMS=np.std(WM, axis=0)
plt.plot(WMM)


#####windrose like a stacked histogram with normed (displayed in percent)

def new_axes(fig, rect):
    ax = WindroseAxes(fig, rect, axisbg='w')
    fig.add_axes(ax)
    return ax


def set_legend(ax):
    l = ax.legend(borderaxespad=-6.8)
    plt.setp(l.get_texts(), fontsize=8)


rect1 = [0.1, 0.1, 0.4, 0.4]
rect2 = [0.6, 0.1, 0.4, 0.4]
rect3 = [1.1, 0.1, 0.4, 0.4]
rect4 = [0.1, -0.45, 0.4, 0.4]
rect5 = [0.6, -0.45, 0.4, 0.4]
rect6 = [1.1, -0.45, 0.4, 0.4]
rect7 = [0.1, -1.05, 0.4, 0.4]
rect8 = [0.6, -1.05, 0.4, 0.4]
rect9 = [1.1, -1.05, 0.4, 0.4]
rect10 = [0.1, -1.6, 0.4, 0.4]
rect11= [0.6, -1.6, 0.4, 0.4]
rect12 = [1.1, -1.6, 0.4, 0.4]


rect= [rect1,rect2,rect3,rect4,rect5,rect6,rect7,rect8,rect9,rect10,rect11,rect12]
namemes=('Enero','Febrero','Marzo','Abril','Mayo','Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre' )

fig = plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='w')

for j,i in enumerate(rect):
    ax1 = new_axes(fig, i)
    ax1.bar(Waves[Waves.index.month==j+1].DIRECCION, Waves[Waves.index.month==j+1].SWH, normed=True, opening=0.8, edgecolor='white')
    set_legend(ax1)
    ax1.set_title(namemes[j],  fontsize=20)
