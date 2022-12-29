#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import sem


# N: número de simulaciones
# n: número de elementos tomados

# In[2]:


N = 10
n = 15


# **SE IMPORTAN LOS DATOS**

# In[3]:


mef_df = pd.read_csv("Data.csv", delimiter=';', decimal=',')
display(mef_df)


# In[4]:


case_df = pd.read_csv("Case.csv", delimiter=';', decimal=',', usecols=[0,1])
display(case_df)


# **MÉTODO DE ELEMENTOS FINITOS**

# Se calcula la tensión media de cada simulación y la desviación estándar experimental de la media de los 8 elementos tomados para cada simulación (Ecuación 4)

# In[5]:


media_simulacion = []
std_simulacion = []
simulaciones = mef_df.columns[1:]
for simulacion in simulaciones:
    media_simulacion.append(mef_df[simulacion].mean())
    std_simulacion.append(sem(mef_df[simulacion])) 
mef_df_2 = pd.DataFrame(media_simulacion, columns = ['Media'])
mef_df_2['Desviacion estándar'] = std_simulacion
display(mef_df_2)


# Se calcula la media (Ecuación 2) y la desviación experimental de la media (Ecuación 3) de las 10 simulaciones efectuadas

# In[6]:


Tension_mef = mef_df_2['Media'].astype('float').mean()
s_mef = sem(mef_df_2['Media'])
Tension_mef


# Se calcula la incertidumbre combinada de la tensión

# In[7]:


u_c = math.sqrt((sum(pow(mef_df_2['Desviacion estándar'].astype('float'),2)) / N) + pow(s_mef,2))
u_c


# Se calcula el factor de cobertura de una distribución tipo t

# In[8]:


V_eff = pow(u_c,4) / ((pow(mef_df_2['Desviacion estándar'].astype('float').mean(),4) / (n-1))+(pow(s_mef,4) / (N-1)))
V_eff


# Interpolamos linealmente entre los 2 valores más cercanos para un intervalo de confianza del 95% (20 - 2.09; 25 - 2.06)

# In[9]:


k_t = (2.06-2.09)/(20-25)*(V_eff - 20) + 2.09
k_t


# La incertidumbre expandida para un nivel de confinza del 95% para la tensión obtenida del modelo de elementos finitos es:

# In[10]:


U_mef = k_t * u_c
U_mef


# **VALIDACIÓN EXPERIMENTAL** (Monte Carlo)

# Se elige el número de reiteraciones de Monte Carlo

# In[11]:


M = 100000
#Coeficiente de expansión térmica del material de ensayo
alpha_B = 0.0000116
#Coeficiente de expansión térmica del material utilizado por el fabricante
alpha_A = 0.0000118


# In[12]:


incertidumbres_df = pd.DataFrame()
ensayos_df = pd.DataFrame()
tolerancia_def = 1
i = 1
for ensayo in case_df['Deformacion']:
    ensayos_df['Ensayo ' + str(i)] = np.random.uniform(low=ensayo-tolerancia_def, high=ensayo+tolerancia_def, size=M)
    i = i+1


# In[13]:


K = 2.07
tolerancia_K = 0.01
incertidumbres_df['Factor de galga'] = np.random.uniform(low=K-tolerancia_K, high=K+tolerancia_K, size=M)


# In[14]:


K_t = 0.001
tolerancia_Kt = 0.001
incertidumbres_df['Factor de sensibilidad transversal'] = np.random.uniform(low=K_t-tolerancia_Kt, high=K_t+tolerancia_Kt, size=M)


# In[15]:


alpha_g = 0.01
tolerancia_alpha = 0.00005
incertidumbres_df['Coeficiente dilatacion termico'] = np.random.uniform(low=alpha_g-tolerancia_alpha, high=alpha_g+tolerancia_alpha, size=M)


# In[16]:


poisson = 0.285
tolerancia_poisson = 0.01
incertidumbres_df['Coeficiente Poisson'] = np.random.uniform(low=poisson-tolerancia_poisson, high=poisson+tolerancia_poisson, size=M)


# In[17]:


Temp_ensayo = 18 #ºC
tolerancia_temp = 1
incertidumbres_df['Temperatura de ensayo'] = np.random.uniform(low=Temp_ensayo-tolerancia_temp, high=Temp_ensayo+tolerancia_temp, size=M)


# In[18]:


Temp_calib = 20 #ºC
incertidumbres_df['Temperatura de calibracion'] = np.random.uniform(low=Temp_calib-tolerancia_temp, high=Temp_calib+tolerancia_temp, size=M)


# In[19]:


F = 14.86 #N
incertidumbre_F = 0.00829
incertidumbres_df['Carga aplicada'] = np.random.normal(loc=F, scale=incertidumbre_F, size=M)


# In[20]:


epsilon = -31.8 + 2.77*Temp_ensayo - 0.0655*pow(Temp_ensayo,2) + 0.000328*pow(Temp_ensayo,3) - 0.000000326*pow(Temp_ensayo,4)
tolerancia_def_ap = 1.7
incertidumbres_df['Deformación aparente'] = np.random.uniform(low=epsilon-tolerancia_def_ap, high=epsilon+tolerancia_def_ap, size=M)


# In[21]:


G = 211 #GPa
incertidumbre_G = 2.11
incertidumbres_df['Modulo Young'] = np.random.normal(loc=G, scale=incertidumbre_G, size=M)


# In[22]:


cita_montaje = 0.02 #rads
tolerancia_cita_montaje = 0.000873
incertidumbres_df['Angulo montaje'] = np.random.uniform(low=cita_montaje-tolerancia_cita_montaje, high=cita_montaje+tolerancia_cita_montaje, size=M)


# ENSAYO 1

# In[23]:


error_no_linealidad = []
error_temperatura = []
error_alineamiento = []
error_sensibilidad_transversal = []
deformacion_corregida = []
tension = []
i = 0
for deformacion in np.array(ensayos_df['Ensayo 1']):
    error_no_linealidad.append(deformacion - (2 * deformacion)/(2 - (0.000001 * deformacion * np.array(incertidumbres_df['Factor de galga'])[i])))
    error_temperatura.append(((2)/(np.array(incertidumbres_df['Factor de galga'])[i] * (1 + np.array(incertidumbres_df['Coeficiente dilatacion termico'])[i] * (np.array(incertidumbres_df['Temperatura de ensayo'])[i] - np.array(incertidumbres_df['Temperatura de calibracion'])[i])))) * (np.array(incertidumbres_df['Deformación aparente'])[i] + (alpha_B - alpha_A)*(np.array(incertidumbres_df['Temperatura de ensayo'])[i] - np.array(incertidumbres_df['Temperatura de calibracion'])[i])))
    error_alineamiento.append(deformacion*(1 - (2)/((1 - np.array(incertidumbres_df['Coeficiente Poisson'])[i]) + (1 + np.array(incertidumbres_df['Coeficiente Poisson'])[i]) * math.cos(np.array(incertidumbres_df['Angulo montaje'])[i]))))
    epsilon_coeff = ((1 - math.cos(2 * np.array(incertidumbres_df['Angulo montaje'])[i])) - np.array(incertidumbres_df['Coeficiente Poisson'])[i] * (1 + math.cos(2 * np.array(incertidumbres_df['Angulo montaje'])[i]))) / ((1 + math.cos(2 * np.array(incertidumbres_df['Angulo montaje'])[i])) - np.array(incertidumbres_df['Coeficiente Poisson'])[i] * (1 - math.cos(2 * np.array(incertidumbres_df['Angulo montaje'])[i])))
    error_sensibilidad_transversal.append((np.array(incertidumbres_df['Factor de sensibilidad transversal'])[i] * (epsilon_coeff + poisson)) / (1 - poisson*np.array(incertidumbres_df['Factor de sensibilidad transversal'])[i]))
    error = error_no_linealidad[i] + error_temperatura[i] + error_alineamiento[i] + error_sensibilidad_transversal[i]
    deformacion_corregida.append(deformacion - error)
    tension.append(deformacion_corregida[i]* 0.000001 * np.array(incertidumbres_df['Modulo Young'])[i] * 1000)
    i = i + 1
ensayo1_df = pd.DataFrame()
ensayo1_df['Deformacion'] = ensayos_df['Ensayo 1']
ensayo1_df['Error No Linealidad'] = error_no_linealidad
ensayo1_df['Error Temperatura'] = error_temperatura
ensayo1_df['Error Alineamiento'] = error_alineamiento
ensayo1_df['Error Sensibilidad Transversal'] = error_sensibilidad_transversal
ensayo1_df['Deformacion corregida'] = deformacion_corregida
ensayo1_df['Tension'] = tension
ensayo1_df.head()


# ENSAYO 2

# In[24]:


error_no_linealidad = []
error_temperatura = []
error_alineamiento = []
error_sensibilidad_transversal = []
deformacion_corregida = []
tension = []
i = 0
for deformacion in np.array(ensayos_df['Ensayo 2']):
    error_no_linealidad.append(deformacion - (2 * deformacion)/(2 - (0.000001 * deformacion * np.array(incertidumbres_df['Factor de galga'])[i])))
    error_temperatura.append(((2)/(np.array(incertidumbres_df['Factor de galga'])[i] * (1 + np.array(incertidumbres_df['Coeficiente dilatacion termico'])[i] * (np.array(incertidumbres_df['Temperatura de ensayo'])[i] - np.array(incertidumbres_df['Temperatura de calibracion'])[i])))) * (np.array(incertidumbres_df['Deformación aparente'])[i] + (alpha_B - alpha_A)*(np.array(incertidumbres_df['Temperatura de ensayo'])[i] - np.array(incertidumbres_df['Temperatura de calibracion'])[i])))
    error_alineamiento.append(deformacion*(1 - (2)/((1 - np.array(incertidumbres_df['Coeficiente Poisson'])[i]) + (1 + np.array(incertidumbres_df['Coeficiente Poisson'])[i]) * math.cos(np.array(incertidumbres_df['Angulo montaje'])[i]))))
    epsilon_coeff = ((1 - math.cos(2 * np.array(incertidumbres_df['Angulo montaje'])[i])) - np.array(incertidumbres_df['Coeficiente Poisson'])[i] * (1 + math.cos(2 * np.array(incertidumbres_df['Angulo montaje'])[i]))) / ((1 + math.cos(2 * np.array(incertidumbres_df['Angulo montaje'])[i])) - np.array(incertidumbres_df['Coeficiente Poisson'])[i] * (1 - math.cos(2 * np.array(incertidumbres_df['Angulo montaje'])[i])))
    error_sensibilidad_transversal.append((np.array(incertidumbres_df['Factor de sensibilidad transversal'])[i] * (epsilon_coeff + poisson)) / (1 - poisson*np.array(incertidumbres_df['Factor de sensibilidad transversal'])[i]))
    error = error_no_linealidad[i] + error_temperatura[i] + error_alineamiento[i] + error_sensibilidad_transversal[i]
    deformacion_corregida.append(deformacion - error)
    tension.append(deformacion_corregida[i]* 0.000001 * np.array(incertidumbres_df['Modulo Young'])[i] * 1000)
    i = i + 1
ensayo2_df = pd.DataFrame()
ensayo2_df['Deformacion'] = ensayos_df['Ensayo 1']
ensayo2_df['Error No Linealidad'] = error_no_linealidad
ensayo2_df['Error Temperatura'] = error_temperatura
ensayo2_df['Error Alineamiento'] = error_alineamiento
ensayo2_df['Error Sensibilidad Transversal'] = error_sensibilidad_transversal
ensayo2_df['Deformacion corregida'] = deformacion_corregida
ensayo2_df['Tension'] = tension
ensayo2_df.head()


# In[25]:


error_no_linealidad = []
error_temperatura = []
error_alineamiento = []
error_sensibilidad_transversal = []
deformacion_corregida = []
tension = []
i = 0
for deformacion in np.array(ensayos_df['Ensayo 3']):
    error_no_linealidad.append(deformacion - (2 * deformacion)/(2 - (0.000001 * deformacion * np.array(incertidumbres_df['Factor de galga'])[i])))
    error_temperatura.append(((2)/(np.array(incertidumbres_df['Factor de galga'])[i] * (1 + np.array(incertidumbres_df['Coeficiente dilatacion termico'])[i] * (np.array(incertidumbres_df['Temperatura de ensayo'])[i] - np.array(incertidumbres_df['Temperatura de calibracion'])[i])))) * (np.array(incertidumbres_df['Deformación aparente'])[i] + (alpha_B - alpha_A)*(np.array(incertidumbres_df['Temperatura de ensayo'])[i] - np.array(incertidumbres_df['Temperatura de calibracion'])[i])))
    error_alineamiento.append(deformacion*(1 - (2)/((1 - np.array(incertidumbres_df['Coeficiente Poisson'])[i]) + (1 + np.array(incertidumbres_df['Coeficiente Poisson'])[i]) * math.cos(np.array(incertidumbres_df['Angulo montaje'])[i]))))
    epsilon_coeff = ((1 - math.cos(2 * np.array(incertidumbres_df['Angulo montaje'])[i])) - np.array(incertidumbres_df['Coeficiente Poisson'])[i] * (1 + math.cos(2 * np.array(incertidumbres_df['Angulo montaje'])[i]))) / ((1 + math.cos(2 * np.array(incertidumbres_df['Angulo montaje'])[i])) - np.array(incertidumbres_df['Coeficiente Poisson'])[i] * (1 - math.cos(2 * np.array(incertidumbres_df['Angulo montaje'])[i])))
    error_sensibilidad_transversal.append((np.array(incertidumbres_df['Factor de sensibilidad transversal'])[i] * (epsilon_coeff + poisson)) / (1 - poisson*np.array(incertidumbres_df['Factor de sensibilidad transversal'])[i]))
    error = error_no_linealidad[i] + error_temperatura[i] + error_alineamiento[i] + error_sensibilidad_transversal[i]
    deformacion_corregida.append(deformacion - error)
    tension.append(deformacion_corregida[i]* 0.000001 * np.array(incertidumbres_df['Modulo Young'])[i] * 1000)
    i = i + 1
ensayo3_df = pd.DataFrame()
ensayo3_df['Deformacion'] = ensayos_df['Ensayo 1']
ensayo3_df['Error No Linealidad'] = error_no_linealidad
ensayo3_df['Error Temperatura'] = error_temperatura
ensayo3_df['Error Alineamiento'] = error_alineamiento
ensayo3_df['Error Sensibilidad Transversal'] = error_sensibilidad_transversal
ensayo3_df['Deformacion corregida'] = deformacion_corregida
ensayo3_df['Tension'] = tension
ensayo3_df.head()


# In[26]:


tension_df = pd.DataFrame()
tension_df['Ensayo 1'] = np.array(ensayo1_df['Tension'])
tension_df['Ensayo 2'] = np.array(ensayo2_df['Tension'])
tension_df['Ensayo 3'] = np.array(ensayo2_df['Tension'])
tensiones = []
i = 0
for tension in tension_df['Ensayo 1']:
    tensiones.append((tension + np.array(tension_df['Ensayo 2'])[i] + np.array(tension_df['Ensayo 3'])[i])/3)
tension_df['Tension media'] = tensiones


# TENSION MEDIA EXPERIMENTAL

# In[32]:


Tension_MMC = tension_df['Tension media'].astype('float').mean()
s_MMC = tension_df['Tension media'].std()
Tension_MMC


# INCERTIDUMBRE EXPANDIDA EXPERIMENTAL

# In[31]:


s_MMC


# In[28]:


plt.hist(tensiones, bins=100, histtype='step')
plt.hist(np.random.normal(loc=Tension_mef, scale=U_mef, size=M), bins=100, histtype='step')
plt.xlim(25,35)
plt.ylim(0, 5000)


# Comprobar correlación del Método de Elementos Finitos con los resultados experimentales de tensión. Si E_N < 1 significa que el MEF repodruce el resultado experimental

# In[29]:


E_N = abs(Tension_mef - Tension_MMC)/ math.sqrt(pow(U_mef,2) - pow(s_MMC,2))
E_N


# El MEF no repodruce fielmente el resultado experimental
