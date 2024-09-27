#%% Librerias
import os
from uncertainties import ufloat, unumpy
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq
#%% FUNCIONES
def cargar_archivo_autom(path):
    # Leer el archivo, omitir la primera fila, y cargar los datos en columnas
    df = pd.read_csv(path, sep='\s+', 
                     skiprows=2, 
                     names=["t", "CH1", "CH2"])
    
    # Convertir las columnas en numpy arrays
    t = df['t'].to_numpy()
    CH1 = df['CH1'].to_numpy()
    CH2 = df['CH2'].to_numpy()
    
    return t, CH1, CH2

def sinusoide(t,A,f,phi,B):
    return A*np.sin(2*np.pi*f*t-phi)+B
#%% Test
path_test = os.path.join(os.getcwd(),'N1_desplazada_40mV','135','135kHz_150dA_100Mss_bobN1Pmag0000.txt')
t, CH1, CH2 = cargar_archivo_autom(path_test)

(A2,f2,phi2,B2),_=curve_fit(sinusoide,t,CH2,p0=[1, 135e3, 0,0])
y2=sinusoide(t,A2,f2,phi2,B2)

def sinusoide2(t,A,phi,B):
    return A*np.sin(2*np.pi*136655.60*t-phi)+B
(A1,phi1,B1),_=curve_fit(sinusoide2,t,CH1,p0=[1,0,0])
y1=sinusoide2(t,A1,phi1,B1)

fig,(ax,ax3)=plt.subplots(nrows=2,figsize=(10,6),constrained_layout=True,sharex=True)
l1,=ax.plot(t,CH1,label='CH1')
ax2=ax.twinx()
l2,=ax2.plot(t,CH2,c='tab:orange',label='CH2')

l3,=ax3.plot(t,y2,c='tab:orange',label='CH2')
ax4=ax3.twinx()
l4,=ax4.plot(t,y1,label='CH1')

lines = [l1, l2]
ax.legend(lines, [line.get_label() for line in lines])

lines2=[l3,l4]
ax3.legend(lines2, [line.get_label() for line in lines2])

ax.grid(axis='x')
ax3.grid(axis='x')
plt.suptitle('N=1 - 135 kHz - 57 kA/m')
plt.show()

#%%

#print(A2,f2,phi2,B2)
#%%



#%%
Phi=phi1-phi2
Tau=np.tan(Phi)/(2*np.pi*f1) 
print('Fiteando sinusoides')
print(f'defasaje ={Phi:.3e} rad')
print(f'tau ={Tau*1e9:.1f} ns')


fig, (ax1,ax2) = plt.subplots(nrows=2,figsize=(10,4),constrained_layout=True,sharex=True)

l1,=ax1.plot(t, CH1, '-', label='CH1')
l12,=ax1.plot(t, y1, '--',c='tab:green', label='Ajuste')

ax1.set_ylabel('Amplitud CH1')
#ax1.set_xlim(0,3/135e3)
ax1.set_ylim(-0.05,0.05)
ax11=ax1.twinx()
l11,=ax11.plot(t, CH2, '-',color='tab:orange', label='CH2')

l2,=ax2.plot(t, y1/max(y1),'-', label='CH1')
ax22=ax2.twinx()
l22,=ax22.plot(t,y2/max(y2) ,'-',color='tab:orange', label='CH2')

ax2.set_xlabel('t (s)')
lines = [l1, l11,l12]
ax1.legend(lines, [line.get_label() for line in lines])
lines2=[l2,l22]
ax2.legend(lines, [line.get_label() for line in lines2])
ax2.grid(axis='x')
plt.show()

fig2, ax = plt.subplots(constrained_layout=True)
ax.plot(y2,y1*1e3)
plt.grid()
plt.xlim(-10,10)
# plt.ylim(-0.03,0.03)
plt.xlabel('CH2 (V)')
plt.ylabel('CH1 (mV)')
plt.show()
#%% FFT (fail)
# Parámetros de la FFT
freq_1 = rfftfreq(len(CH1), d=1e-8)  # Frecuencias usando rfft
g_1_aux = rfft(CH1)         # Transformada de Fourier de la señal CH1
mag_1 = abs(g_1_aux)        # Magnitud de la FFT
fase_1 = np.angle(g_1_aux)

freq_2 = rfftfreq(len(CH2), d=1e-8)  # Frecuencias usando rfft
g_2_aux = rfft(CH2)         # Transformada de Fourier de la señal CH2
mag_2 = abs(g_2_aux)        # Magnitud de la FFT
fase_2 = np.angle(g_2_aux)

# Filtrar las frecuencias según el límite

frec_limite = 300e3 * 5  # Limitar la frecuencia máxima en el gráfico de FFT
mask_frecuencia = freq_1 <= frec_limite
freq_1 = freq_1[mask_frecuencia]
mag_1 = mag_1[mask_frecuencia]
fase_1 = fase_1[mask_frecuencia]

freq_2 = freq_2[mask_frecuencia]
mag_2 = mag_2[mask_frecuencia]
fase_2 = fase_2[mask_frecuencia]

dphi=max(fase_1)-max(fase_2)
tau= np.tan(dphi)/(2*np.pi*300e3)
print('FFT')
print(f'defasaje ={dphi:.3e} rad')
print(f'tau ={tau*1e9:.1f} ns')

# Crear la figura y el eje principal
fig, (ax1,ax2) = plt.subplots(nrows=2,figsize=(10,4),constrained_layout=True)

# Ploteo del primer par de datos (ejemplo de x, y)
l1,=ax1.plot(t, CH1, '-', label='CH1')
ax1.set_xlabel('Tiempo (s)')
ax1.set_ylabel('Amplitud CH1')
ax1.set_xlim(0,3/300e3)  # Limitar el rango de frecuencias visibles

ax11=ax1.twinx()
l11,=ax11.plot(t, CH2, '-',color='tab:orange', label='CH2')
# Ploteo del espectro de CH2 (frecuencia vs magnitud)

l2,=ax2.plot(freq_1, mag_1,'.-', label='CH1 (FFT)')
#ax2.plot(freq_1[np.nonzero(mag_1==max(mag_1))],max(mag_1),'o')
ax2.set_ylabel('Magnitud CH1')
ax2.set_xlabel('Frecuencia')
#ax2.set_xlim([0, frec_limite])  # Limitar el rango de frecuencias visibles
ax22=ax2.twinx()
l22,=ax22.plot(freq_2, mag_2,'.-',color='tab:orange', label='CH2 (FFT)')

lines = [l1, l11]
ax1.legend(lines, [line.get_label() for line in lines])
lines2=[l2,l22]
ax2.legend(lines, [line.get_label() for line in lines])
ax2.grid(axis='x')
plt.show()


#%% 300 
dphi_300_1=[]
tau_300_1=[]
path_300_1 = glob('N1_desplazada_40mV/300/300kHz*.txt')
path_300_1.sort()

for f in path_300_1:
    path = os.path.join(os.getcwd(),f)
    t, CH1, CH2 = cargar_archivo_autom(path)

    (A2,f2,phi2,B2),_=curve_fit(sinusoide,t,CH2,p0=[1, 300e3, 0,0])
    y2=sinusoide(t,A2,f2,phi2,B2)
    
    def sinusoide2(t,A,phi,B):
        return A*np.sin(2*np.pi*f2*t-phi)+B
    (A1,phi1,B1),_=curve_fit(sinusoide2,t,CH1,p0=[1,0,0])
    y1=sinusoide2(t,A1,phi1,B1)
    
    
    Phi=phi1-phi2
    Tau=np.tan(Phi)/(2*np.pi*f2) 
    dphi_300_1.append(np.mod(Phi,2*np.pi))
    tau_300_1.append(Tau)

    print(f'defasaje ={Phi:.3e} rad (mod 2pi={np.mod(Phi,2*np.pi):.3e} rad)')
    print(f'tau ={Tau*1e9:.1f} ns')

    fig,(ax,ax3)=plt.subplots(nrows=2,figsize=(10,6),constrained_layout=True,sharex=True)
    l1,=ax.plot(t,CH1,label='CH1')
    ax2=ax.twinx()
    l2,=ax2.plot(t,CH2,c='tab:orange',label='CH2')
    l3,=ax3.plot(t,y2,c='tab:orange',label='CH2')
    ax4=ax3.twinx()
    l4,=ax4.plot(t,y1,label='CH1')
    lines = [l1, l2]
    ax.legend(lines, [line.get_label() for line in lines])
    lines2=[l3,l4]
    ax3.legend(lines2, [line.get_label() for line in lines2])
    ax.grid(axis='x')
    ax3.grid(axis='x')
    plt.suptitle('N=1 - 300 kHz - 57 kA/m \n'+f.split('/')[-1])
    plt.show()

print('-'*50)

Dphi_300=ufloat(np.mean(dphi_300_1),np.std(dphi_300_1))

print(f'''Dphi 300 kHz = {Dphi_300:.1e} rad = {ufloat(np.degrees(Dphi_300.nominal_value),np.degrees(Dphi_300.std_dev)):.2f}°  -  ({len(dphi_300_1)} muestras)
''')
Tau_300 = ufloat(np.tan(Dphi_300.nominal_value)/(2*np.pi*300e3),np.tan(Dphi_300.std_dev)/(2*np.pi*300e3))

print(f'''Tau 300 kHz = {Tau_300*1e9:.1f} ns''')
print('-'*50)

#%% 135

dphi_135_1=[]
tau_135_1=[]
path_135_1 = glob('N1_desplazada_40mV/135/135kHz*.txt')
path_135_1.sort()

for f in path_135_1:
    path = os.path.join(os.getcwd(),f)
    t, CH1, CH2 = cargar_archivo_autom(path)

    (A2,f2,phi2,B2),_=curve_fit(sinusoide,t,CH2,p0=[1, 135e3, 0,0])
    y2=sinusoide(t,A2,f2,phi2,B2)
    
    def sinusoide2(t,A,phi,B):
        return A*np.sin(2*np.pi*f2*t-phi)+B
    (A1,phi1,B1),_=curve_fit(sinusoide2,t,CH1,p0=[1,0,0])
    y1=sinusoide2(t,A1,phi1,B1)
    
    
    Phi=phi1-phi2
    Tau=np.tan(Phi)/(2*np.pi*f2) 
    dphi_135_1.append(np.mod(Phi,2*np.pi))
    tau_135_1.append(Tau)

    print(f'defasaje ={Phi:.3e} rad (mod 2pi={np.mod(Phi,2*np.pi):.3e} rad)')
    print(f'tau ={Tau*1e9:.1f} ns')

    fig,(ax,ax3)=plt.subplots(nrows=2,figsize=(10,6),constrained_layout=True,sharex=True)
    l1,=ax.plot(t,CH1,label='CH1')
    ax2=ax.twinx()
    l2,=ax2.plot(t,CH2,c='tab:orange',label='CH2')
    l3,=ax3.plot(t,y2,c='tab:orange',label='CH2')
    ax4=ax3.twinx()
    l4,=ax4.plot(t,y1,label='CH1')
    lines = [l1, l2]
    ax.legend(lines, [line.get_label() for line in lines])
    lines2=[l3,l4]
    ax3.legend(lines2, [line.get_label() for line in lines2])
    ax.grid(axis='x')
    ax3.grid(axis='x')
    plt.suptitle('N=1 - 135 kHz - 57 kA/m \n'+f.split('/')[-1])
    plt.show()

print('-'*50)

Dphi_135=ufloat(np.mean(dphi_135_1),np.std(dphi_135_1))

print(f'''Dphi 135 kHz = {Dphi_135:.1e} rad = {ufloat(np.degrees(Dphi_135.nominal_value),np.degrees(Dphi_135.std_dev)):.2f}°  -  ({len(dphi_135_1)} muestras)
''')
Tau_135 = ufloat(np.tan(Dphi_135.nominal_value)/(2*np.pi*f2),np.tan(Dphi_135.std_dev)/(2*np.pi*f2))

print(f'''Tau 135 kHz = {Tau_135*1e9:.1f} ns''')
print('-'*50)

# %%
