#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 22:00:21 2021

@author: jeoa
"""

import gensim
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

@st.cache(persist= True, suppress_st_warning= True)
def carga(archivo):
    datos = pd.read_csv('Datos/' + archivo, index_col= 0)
    return datos

def main():
    st.title('2.1. Caso de estudio: micromecenazgo cultural y creativo en México y América Latina')
    st.write('Describir la aplicación a detalle')
    opcion = st.sidebar.selectbox(label= 'Selecciona una opción',
                                  options= ['Bienvenida', 'Reporte general de datos',
                                            'Análisis Exploratorio (EDA: Exploratory Data Analysis',
                                            'Sistema de recomendación por correlación de métricas',
                                            'Sistema de recomendación basado en contenido'])
    if opcion == 'Reporte general de datos':
        reporte()
    
def reporte():
    opcion_reporte = st.sidebar.selectbox(label= 'Elije uno de los conjuntos de datos disponibles', 
                         options= ['México', 'América Latina'])
    if opcion_reporte == 'México':
        datos = carga(archivo= 'datos_kickstarter_preprocesados.csv')
        st.dataframe(data= datos)
        reporte = ProfileReport(datos, explorative= True)
        if st.button(label= 'Generar reporte'):
            st_profile_report(reporte)
    elif opcion_reporte == 'América Latina':
        datos = carga(archivo= 'datos_ideame_preprocesados.csv')
        st.dataframe(data= datos)
        reporte = ProfileReport(datos, explorative= True)
        if st.button(label= 'Generar reporte'):
            st_profile_report(reporte)
        
if __name__ == '__main__':
    main()
    
















