#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jesús Eduardo Oliva Abarca
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sn
import texthero as hero
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(persist= True, suppress_st_warning= True)
def carga(archivo):
    datos = pd.read_csv('Datos/' + archivo, index_col= 0)
    return datos

datos_kickstarter = carga('datos_kickstarter_preprocesados.csv')

tfidf_datos_kickstarter = TfidfVectorizer()
matriz_tfidf_kickstarter = tfidf_datos_kickstarter.fit_transform(datos_kickstarter['Descripción (limpia)'])
coseno_similitud_kickstarter = linear_kernel(matriz_tfidf_kickstarter, matriz_tfidf_kickstarter)
indices_kickstarter = pd.Series(data= datos_kickstarter.index, index= datos_kickstarter['Nombre del proyecto'])

datos_ideame = carga('datos_ideame_preprocesados.csv')

tfidf_datos_ideame = TfidfVectorizer()
matriz_tfidf_ideame = tfidf_datos_ideame.fit_transform(datos_ideame['Descripción (limpia)'])
coseno_similitud_ideame = linear_kernel(matriz_tfidf_ideame, matriz_tfidf_ideame)
indices_ideame = pd.Series(data= datos_ideame.index, index= datos_ideame['Proyecto'])

def obtener_recomendacion_kickstarter(proyecto, coseno= coseno_similitud_kickstarter):
    idx = indices_kickstarter[proyecto]
    puntajes_similitud = list(enumerate(coseno_similitud_kickstarter[idx]))
    puntajes_similitud = sorted(puntajes_similitud, key= lambda x: x[1], reverse= True)
    puntajes_similitud = puntajes_similitud[1:11]
    indices_proyectos = [i[0] for i in puntajes_similitud]
    return datos_kickstarter[['Nombre del proyecto', 'Categoría', 'Estatus', 'Palabras clave', 'Recaudado', 
                                                 'Objetivo', 'Conteo de patrocinadores']].iloc[indices_proyectos].sort_values(by= 'Conteo de patrocinadores', 
                                                                                                                  ascending= False)
                                                                                                                 
def obtener_recomendacion_ideame(proyecto, coseno= coseno_similitud_ideame):
    idx = indices_ideame[proyecto]
    puntajes_similitud = list(enumerate(coseno_similitud_ideame[idx]))
    puntajes_similitud = sorted(puntajes_similitud, key= lambda x: x[1], reverse= True)
    puntajes_similitud = puntajes_similitud[1:11]
    indices_proyectos = [i[0] for i in puntajes_similitud]
    return datos_ideame[['Proyecto', 'Categoría', 'Palabras clave','Recaudación', 'Porcentaje', 'Vistas',
                                                 'Colaboradores']].iloc[indices_proyectos].sort_values(by= 'Vistas',
                                                                                                                  ascending= False)


def main():
    st.title('Caso de estudio: micromecenazgo cultural y creativo en México y América Latina')
    opcion = st.sidebar.selectbox(label= 'Selecciona una opción',
                                  options= ['Bienvenida', 'Reporte general de datos',
                                            'Análisis Exploratorio (EDA: Exploratory Data Analysis',
                                            'Sistema de recomendación por correlación de métricas',
                                            'Sistema de recomendación basado en contenido'])
    if opcion == 'Bienvenida':
       bienvenida()
    elif opcion == 'Reporte general de datos':
        reporte()
    elif opcion == 'Análisis Exploratorio (EDA: Exploratory Data Analysis':
        eda()
    elif opcion == 'Sistema de recomendación por correlación de métricas':
        recomendacion_correlacion()
    elif opcion == 'Sistema de recomendación basado en contenido':
        recomendacion_contenido()
        
def bienvenida():
    st.markdown("""Esta aplicación web ha sido desarrollada por Jesús Eduardo Oliva Abarca, como parte de un proyecto general de investigación 
    que parte del enfoque de la analítica cultural de Lev Manovich, el cual aborda las aplicaciones de las herramientas, métodos y técnicas
    de la ciencia de datos para el estudio de conjuntos de datos culturales masivos.
    En esta aplicación, el usuario puede examinar los datos extraídos —a través de técnicas de "raspado de red" (o *web scraping*),
    así como del sitio *Web Robots* (https://webrobots.io/)— de las páginas de *Ideame* (https://www.idea.me/), y de *Kicstarter* 
    (https://www.kickstarter.com/mexico), para identificar patrones y tendencias relativas al micromecenazgo cultural y creativo en
    México y América Latina.""")
    st.markdown("""El propósito de esta aplicación es ofrecer a las interesadas e interesados una herramienta de análisis para la toma de
                decisiones en lo que respecta a las posibilidades de financiamiento de sus proyectos culturales y creativos, así como en
                lo relativo a las categorías y temas en los que exista una mayor probabilidad de éxito.""")
    st.markdown("""Es importante indicar, que para la obtención de los conjuntos de datos en los que se basa esta aplicación, se respetaron
    los protocolos de estándar de exclusión de robots que aparecen en los archivos "robots.txt" de cada página. Además, es 
    pertinente indicar el reconocimiento a las páginas antes citadas, que son sitios confiables y seguros, dedicados a la financiación 
    de proyectos de diverso tipo. También, se reconoce el esfuerzo de creadoras y creadores, promotores y promotoras, gestoras y gestores, 
    y demás personas involucradas en la cultura, las artes y la creatividad, que optan por la alternativa del micromecenazgo para promover 
    y continuar con sus actividades.""")
    st.markdown("""Los datos empleados para esta aplicación son tratados con todo respeto y confidencialidad, y se solicita a todo usuario su
    apoyo para promover tanto las páginas de *Ideame* y *Kicstarter*, entre otras más dedicadas a la financiación colectiva, así como
    para difundir el trabajo de artistas, productores, diseñadores y demás profesionales de la cultura y de la creatividad.
    Un último aspecto a señalar es que ambos conjuntos de datos corresponden a los proyectos existentes en ambos sitios hasta el mes de
    septiembre de 2020.
    Cualquier duda o comentario: 
        
    jeduardo.oliv@gmail.com""")
    
    st.markdown('https://github.com/JEOA-1981')
    st.markdown('https://www.linkedin.com/in/jes%C3%BAs-eduardo-oliva-abarca-78b615157/')
    st.markdown('#Nota: esta aplicación se encuentra aún en fase de desarrollo, su uso recomendado es meramente como una herramienta de análisis')
    
def reporte():
    with st.beta_expander(label= 'Descripción de la sección', expanded= True):
        st.subheader('En esta sección, el usuario puede conocer las características generales de los dos conjuntos de datos disponibles, ya sea mediante una descripción general de cada uno, visualizándolos como tabla de datos (con filas y columnas), o a través de la generación de un reporte con información detallada de cada variable.')
    opcion_reporte = st.selectbox(label= 'Seleciona uno de los conjuntos de datos disponibles', 
                         options= ['México', 'América Latina'])
    seleccion = st.sidebar.radio(label= 'Escoge una opción', options= ['Descripción general', 'Tabla de datos', 'Reporte general'])
    if seleccion == 'Descripción general':
        if opcion_reporte == 'México':
            datos = carga(archivo= 'datos_kickstarter_preprocesados.csv')
            st.dataframe(data= datos)
            st.subheader('Este conjunto de datos cuenta con las siguientes características generales:')
            st.info("""Cuenta con 11 variables, de las cuales 8 son categóricas (Nombre del proyecto, Creador(a), 
            Estatus, Ubicación (por estado), Descripción, Descripción (limpia) —descripción de cada proyecto 
            luego del preprocesamiento de texto—, Palabras clave y Categoría) y 3 son numéricas (Recaudado (monto en pesos mexicanos), 
            Objetivo, y Conteo de patrocinadores). Además de 1842 observaciones, sin valores nulos ni duplicados (se realizó un 
            preprocesamiento de los datos y varias comprobaciones para su mejor legibilidad)""")
        elif opcion_reporte == 'América Latina':
            datos = carga(archivo= 'datos_ideame_preprocesados.csv')
            st.dataframe(data= datos)
            st.subheader('Las características generales de este conjunto de datos son las siguientes:')
            st.info("""Cuenta con 11 variables, de las cuales 7 son categóricas (Proyecto, Autor(a), País, Descripción, 
            Descripción (limpia) —descripción de cada proyecto luego del preprocesamiento de texto—, Palabras clave y Categoría) y 
            4 son numéricas (Recaudación (en pesos mexicanos), Porcentaje, Vistas, Colaboradores). Además de 1824 observaciones, sin 
            valores nulos ni duplicados (se realizó un preprocesamiento de los datos y varias comprobaciones para su mejor legibilidad)""")
    if seleccion == 'Tabla de datos':
        if opcion_reporte == 'México':
            datos = carga(archivo= 'datos_kickstarter_preprocesados.csv')
            st.info('Tabulación cruzada: número de proyectos por país')
            st.dataframe(pd.crosstab(index= datos['Ubicación (por estado)'], columns= datos['Categoría']))
            st.info('Matriz de correlaciones')
            st.dataframe(data= datos.corr())
            st.info('Métricas/estadísticas generales')
            st.dataframe(data= datos.describe())
        elif opcion_reporte == 'América Latina':
            datos = carga(archivo= 'datos_ideame_preprocesados.csv')
            st.info('Tabulación cruzada: número de proyectos por país')
            st.dataframe(pd.crosstab(index= datos['País'], columns= datos['Categoría']))
            st.info('Matriz de correlaciones')
            st.dataframe(data= datos.corr())
            st.info('Métricas/estadísticas generales')
            st.dataframe(data= datos.describe())
    if seleccion == 'Reporte general':
        if opcion_reporte == 'México':
            datos = carga(archivo= 'datos_kickstarter_preprocesados.csv')
            reporte = ProfileReport(datos, explorative= True)
            if st.button(label= 'Generar reporte'):
                st_profile_report(reporte)
        elif opcion_reporte == 'América Latina':
            datos = carga(archivo= 'datos_ideame_preprocesados.csv')
            reporte = ProfileReport(datos, explorative= True)
            if st.button(label= 'Generar reporte'):
                st_profile_report(reporte)
                
def eda():
    with st.beta_expander(label= 'Descripción de la sección', expanded= True):
        st.subheader('El análisis exploratorio de datos (*Exploratory Data Analysis*, o EDA), consiste en un examen de los atributos y características generales de un conjunto de datos, recurriendo a la exploración de sus métricas, o a la visualización de los datos mediante diferentes tipos de gráficos. En esta sección, el usuario podrá explorar los conjuntos de datos disponibles mediante la visualización de las métricas correspondientes, además de poder examinar los datos agrupados según espacio geopolítico (país, en América Latina, estado de la república, en México) y la categoría cultural que desea explorar.')
    opcion_reporte = st.selectbox(label= 'Selecciona uno de los conjuntos de datos disponibles', 
                         options= ['México', 'América Latina'])
    if opcion_reporte == 'México':
        datos = carga(archivo= 'datos_kickstarter_preprocesados.csv')
        
        st.subheader('Los gráficos muestran las métricas correspondientes a todo el conjunto de datos')
        opcion_metrica = st.radio('Selecciona', options= ['Recaudado', 'Objetivo', 'Conteo de patrocinadores']) 
        
        recaudado, objetivo, conteo_patrocinadores = st.beta_columns(3)
        
        
        boton_uno, boton_dos = st.beta_columns(2)
        with boton_uno:
            if st.checkbox(label= 'Generar gráfico por categoría'):
                st.info('Maximiza el gráfico para una mejor visualización')
                plt.figure(figsize= (200, 60))
                fig, ax = plt.subplots()
                ax = sn.barplot(data= datos, x= opcion_metrica, y= 'Ubicación (por estado)', hue= 'Categoría', ci= None)
                plt.gca().ticklabel_format(axis= 'x', style= 'plain', useOffset= False)
                plt.xticks(rotation= 45)
                plt.title('Correlación de proyectos por estado de la república según categoría')
                st.pyplot(fig)
        with boton_dos:
            if st.checkbox(label= 'Generar gráfico por estatus del proyecto', key= 1):
                st.info('Maximiza el gráfico para una mejor visualización')
                plt.figure(figsize= (200, 60))
                fig, ax = plt.subplots()
                ax = sn.barplot(data= datos, x= opcion_metrica, y= 'Ubicación (por estado)', hue= 'Estatus', ci= None)
                plt.gca().ticklabel_format(axis= 'x', style= 'plain', useOffset= False)
                plt.xticks(rotation= 45)
                plt.title('Correlación de proyectos por estado de la república según estatus')
                st.pyplot(fig)
        
        st.subheader('La tabla que se muestra aquí corresponde a la agrupación seleccionada según los criterios elegidos en la barra lateral')
        estados = datos['Ubicación (por estado)'].unique()
        estado = st.sidebar.selectbox(label= 'Selecciona un estado', options= estados)
        categorias = datos['Categoría'].unique()
        categoria = st.sidebar.radio(label= 'Seleccionar la categoría', options= categorias) 
        try:
            if categoria in datos['Categoría'].values:
                seleccion_datos = datos.groupby(by= ['Ubicación (por estado)', 'Categoría']).get_group((estado, categoria))
                st.dataframe(data= seleccion_datos)
                st.info('El total de proyectos de ' + categoria.lower() + ' es de ' + str(len(seleccion_datos.index)))
                if st.button(label= 'Generar gráfico', key= 2):
                    st.info('Maximiza el gráfico para una mejor visualización')
                    plt.figure(figsize= (200, 60))
                    fig, ax = plt.subplots()
                    ax = sn.pointplot(data= seleccion_datos, x= 'Recaudado', y= 'Objetivo', hue= 'Estatus')
                    plt.gca().ticklabel_format(axis= 'y', style= 'plain', useOffset= False)
                    plt.xticks(rotation= 45, fontsize= 6)
                    plt.title('Relaciones entre niveles de las variables seleccionadas')
                    st.pyplot(fig)
                    st.markdown('El gráfico permite examinar el éxito de un proyecto (basado en la recaudación y el objetivo planteado) según su categoría')
        except KeyError:
            st.warning('No existen proyectos de esta categoría')
          
    elif opcion_reporte == 'América Latina':
        datos = carga(archivo= 'datos_ideame_preprocesados.csv')
        
        st.subheader('Los gráficos muestran las métricas correspondientes a todo el conjunto de datos')

        st.subheader('Selecciona la métrica para el eje y del gráfico')
        recaudacion, porcentaje, vistas, colaboradores = st.beta_columns(4)
        with recaudacion:
            if st.checkbox(label= 'Recaudación'):
                plt.figure(figsize= (60, 30))
                fig, ax = plt.subplots()
                ax = sn.barplot(data= datos, x= 'País', y= 'Recaudación', hue= 'Categoría')
                plt.gca().ticklabel_format(axis= 'y', style= 'plain', useOffset= False)
                plt.xticks(rotation= 45)
                plt.title('Correlación de proyectos por país según categoría')
                st.pyplot(fig)
        with porcentaje:
            if st.checkbox(label= 'Porcentaje', key= 1):
                plt.figure(figsize= (60, 30))
                fig, ax = plt.subplots()
                ax = sn.barplot(data= datos, x= 'País', y= 'Porcentaje', hue= 'Categoría')
                plt.gca().ticklabel_format(axis= 'y', style= 'plain', useOffset= False)
                plt.xticks(rotation= 45)
                plt.title('Correlación de proyectos por país según categoría')
                st.pyplot(fig)
        with vistas:
            if st.checkbox(label= 'Vistas', key= 2):
                plt.figure(figsize= (60, 30))
                fig, ax = plt.subplots()
                ax = sn.barplot(data= datos, x= 'País', y= 'Vistas', hue= 'Categoría')
                plt.gca().ticklabel_format(axis= 'y', style= 'plain', useOffset= False)
                plt.xticks(rotation= 45)
                plt.title('Correlación de proyectos por país según categoría')
                st.pyplot(fig)
        with colaboradores:
            if st.checkbox(label= 'Colaboradores', key= 3):
                plt.figure(figsize= (60, 30))
                fig, ax = plt.subplots()
                ax = sn.barplot(data= datos, x= 'País', y= 'Colaboradores', hue= 'Categoría')
                plt.gca().ticklabel_format(axis= 'y', style= 'plain', useOffset= False)
                plt.xticks(rotation= 45)
                plt.title('Correlación de proyectos por país según categoría')
                st.pyplot(fig)
        st.info('Maximiza los gráficos para una mejor visualización')
        
        st.subheader('La tabla que se muestra aquí corresponde a la agrupación seleccionada según los criterios elegidos en la barra lateral')
        paises = datos['País'].unique()
        pais = st.sidebar.selectbox(label= 'Selecciona un país', options= paises)
        categorias = datos['Categoría'].unique()
        categoria = st.sidebar.radio(label= 'Seleccionar la categoría', options= categorias) 
        try:
            if categoria in datos['Categoría'].values:
                seleccion_datos = datos.groupby(by= ['País', 'Categoría']).get_group((pais, categoria))
                st.dataframe(data= seleccion_datos)
                st.info('El total de proyectos de ' + categoria.lower() + ' es de ' + str(len(seleccion_datos.index)))
                
                st.subheader('Selecciona la métrica para el eje y del gráfico')
                porcentaje, vistas, colaboradores = st.beta_columns(3)
                with porcentaje:
                    if st.checkbox(label= 'Porcentaje'):
                        plt.figure(figsize= (200, 60))
                        fig, ax = plt.subplots()
                        ax = sn.pointplot(data= seleccion_datos, x= 'Recaudación', y= 'Porcentaje')
                        plt.gca().ticklabel_format(axis= 'y', style= 'plain', useOffset= False)
                        plt.xticks(rotation= 45, fontsize = 6)
                        plt.title('Relaciones entre las variables seleccionadas')
                        st.pyplot(fig)
                with vistas:
                    if st.checkbox(label= 'Vistas', key= 1):
                        plt.figure(figsize= (200, 60))
                        fig, ax = plt.subplots()
                        ax = sn.pointplot(data= seleccion_datos, x= 'Recaudación', y= 'Vistas')
                        plt.gca().ticklabel_format(axis= 'y', style= 'plain', useOffset= False)
                        plt.xticks(rotation= 45, fontsize = 6)
                        plt.title('Relaciones entre las variables seleccionadas')
                        st.pyplot(fig)
                with colaboradores:
                    if st.checkbox(label= 'Colaboradores', key= 2):
                        plt.figure(figsize= (200, 60))
                        fig, ax = plt.subplots()
                        ax = sn.pointplot(data= seleccion_datos, x= 'Recaudación', y= 'Colaboradores')
                        plt.gca().ticklabel_format(axis= 'y', style= 'plain', useOffset= False)
                        plt.xticks(rotation= 45, fontsize = 6)
                        plt.title('Relaciones entre las variables seleccionadas')
                        st.pyplot(fig)
                st.markdown("""Los gráficos permiten examinar el éxito de un proyecto (basado en la recaudación, las vistas, los colaboradores
                y el porcentaje de recursos obtenidos) según su categoría""")
                st.info('Maximiza los gráficos para una mejor visualización')
        except KeyError:
           st.warning('No existen proyectos de esta categoría')
              
def recomendacion_correlacion():
    formato = tkr.ScalarFormatter(useMathText= False)
    formato.set_scientific(False)
    with st.beta_expander(label= 'Descripción de la sección', expanded= True):
        st.subheader('Una de las aplicaciones de la ciencia de datos es la del diseño de sistemas de recomendación, los cuales "predicen" las posibles selecciones del usuario basándose en sus preferencias previas, ya sea con base en las correlaciones de las métricas entre productos, servicios o eventos, o bien, por el examen de reseñas o comentarios de los usuarios, o, finalmente, en las interacciones de éstos (valoraciones, puntajes, *ratings*). En esta sección, se le presenta al usuario un sistema de recomendación por correlación de métricas.')
    opcion_reporte = st.selectbox(label= 'Seleciona uno de los conjuntos de datos disponibles', 
                                  options= ['México', 'América Latina'])
    if opcion_reporte == 'México':
            datos = carga(archivo= 'datos_kickstarter_preprocesados.csv')
            metricas_proyectos = datos.copy().groupby(by= ['Nombre del proyecto', 'Categoría', 'Ubicación (por estado)'])[['Recaudado', 'Objetivo', 
                                                'Conteo de patrocinadores']].mean().sort_values(by= 'Objetivo', ascending= False)
            matriz_proyectos = datos.pivot_table(columns= ['Nombre del proyecto'])
            
            nombres_proyectos = datos['Nombre del proyecto'].unique()
            nombre_proyecto = st.sidebar.selectbox(label= 'Selecciona el proyecto', options= nombres_proyectos)
            
            proyectos_similares = matriz_proyectos.corrwith(matriz_proyectos[nombre_proyecto])
            correlacion_proyectos = pd.DataFrame(proyectos_similares, columns= ['Correlación'])
            correlacion_proyectos.dropna(inplace= True)
            correlacion_proyectos = correlacion_proyectos.join(metricas_proyectos)
            st.info('Proyectos correlacionados con el seleccionado')
            st.dataframe(data= correlacion_proyectos.sort_values(by= 'Correlación', ascending= False).head(n= 10))
            
            if st.button(label= 'Generar gráfico'):
                cmap = sn.diverging_palette(230, 20, as_cmap=True)
                plt.figure(figsize= (60, 30))
                fig, ax = plt.subplots()
                ax = sn.heatmap(correlacion_proyectos.sort_values(by= 'Correlación', ascending= False).head(n= 10), cmap= cmap, 
                                linewidths= 3, annot= True, fmt= '.0f', cbar_kws={"format": formato})
                st.pyplot(fig)
                st.markdown("""Los datos y el gráfico mostrados indican los proyectos que con mayor probabilidad se nos recomendarían, 
                            según el proyecto elegido, y con base en la correlación entre las métricas.""")
 
    elif opcion_reporte == 'América Latina':
            datos = carga(archivo= 'datos_ideame_preprocesados.csv')
            metricas_proyectos = datos.copy().groupby(by= ['Proyecto', 'Categoría', 'País'])[['Recaudación', 'Porcentaje', 
                                                'Vistas', 'Colaboradores']].mean().sort_values(by= 'Porcentaje', ascending= False)
            matriz_proyectos = datos.pivot_table(columns= ['Proyecto'])
            
            nombres_proyectos = datos['Proyecto'].unique()
            nombre_proyecto = st.sidebar.selectbox(label= 'Selecciona el proyecto', options= nombres_proyectos)
            
            #st.info('Información del proyecto seleccionado')
            #st.dataframe(data= metricas_proyectos[metricas_proyectos.index.get_level_values('Nombre del proyecto').str.contains(nombre_proyecto)])
            
            proyectos_similares = matriz_proyectos.corrwith(matriz_proyectos[nombre_proyecto])
            correlacion_proyectos = pd.DataFrame(proyectos_similares, columns= ['Correlación'])
            correlacion_proyectos.dropna(inplace= True)
            correlacion_proyectos = correlacion_proyectos.join(metricas_proyectos)
            st.info('Proyectos correlacionados con el seleccionado')
            st.dataframe(data= correlacion_proyectos.sort_values(by= 'Correlación', ascending= False).head(n= 10))
            
            if st.button(label= 'Generar gráfico'):
                cmap = sn.diverging_palette(230, 20, as_cmap=True)
                plt.figure(figsize= (60, 30))
                fig, ax = plt.subplots()
                ax = sn.heatmap(correlacion_proyectos.sort_values(by= 'Correlación', ascending= False).head(n= 10), cmap= cmap,
                                linewidths= 3, annot= True, fmt= '.0f', cbar_kws={"format": formato})
                st.pyplot(fig)
                st.markdown("""Los datos y el gráfico mostrados indican los proyectos que con mayor probabilidad se nos recomendarían, 
                            según el proyecto elegido, y con base en la correlación entre las métricas.""")
    
def recomendacion_contenido():
    formato = tkr.ScalarFormatter(useMathText= False)
    formato.set_scientific(False)
    
    with st.beta_expander(label= 'Descripción de la sección', expanded= True):
        st.subheader('A diferencia de la sección anterior, el sistema de clasificación mostrado aquí se basa tanto en  en las métricas de los proyectos, así como en sus descripciones. Este sistema basado en contenido complementa las correlaciones de las métricas del anterior con los datos textuales empleados que describen o reseñan productos, eventos y servicios. En este caso, la premisa de identificar las similitudes entre los proyectos que tienen más vistas y más colaboradores.')
        
    opcion_reporte = st.selectbox(label= 'Seleciona uno de los conjuntos de datos disponibles', 
                              options= ['México', 'América Latina'])
    if opcion_reporte == 'México':
        datos = carga(archivo= 'datos_kickstarter_preprocesados.csv')
        if st.button(label= 'Generar nube de palabras'):
            nube = hero.wordcloud(datos['Palabras clave'])
            st.pyplot(nube)
            st.markdown('La nube de palabras permite identificar los temas más recurrentes según las palabras clave de todo el conjunto de datos')
            
        nombres_proyectos = datos['Nombre del proyecto'].unique()
        nombre_proyecto = st.sidebar.selectbox(label= 'Selecciona el proyecto', options= nombres_proyectos)
        st.dataframe(data= datos[datos['Nombre del proyecto'].str.contains(nombre_proyecto)])
        st.dataframe(data= obtener_recomendacion_kickstarter(nombre_proyecto))
        
        opcion_uno, opcion_dos = st.beta_columns(2)
        
        with opcion_uno:
            if st.checkbox(label= 'Generar nube de palabras', key= 1):
                nube = hero.wordcloud(obtener_recomendacion_kickstarter(nombre_proyecto)['Palabras clave'])
                st.pyplot(nube)
                st.markdown('La nube de palabras permite identificar los temas más recurrentes según las palabras clave de los proyectos relacionados conforme a sus métricas y descripciones')
                st.info('Maximiza los gráficos para una mejor visualización')
                
        with opcion_dos:
            if st.checkbox(label= 'Generar gráfico', key= 2):
                plt.figure(figsize= (60, 30))
                fig, ax = plt.subplots()
                ax = sn.scatterplot(data= obtener_recomendacion_kickstarter(nombre_proyecto), x= 'Objetivo',
                                y= 'Nombre del proyecto', hue= 'Categoría', style= 'Estatus')
                plt.gca().ticklabel_format(axis= 'x', style= 'plain', useOffset= False)
                plt.xticks(rotation= 45)
                plt.title('Estatus de los proyectos relacionados según el objetivo de recaudación')
                st.pyplot(fig)
                st.info('Maximiza los gráficos para una mejor visualización')
            
    elif opcion_reporte == 'América Latina':
        datos = carga(archivo= 'datos_ideame_preprocesados.csv')
        if st.button(label= 'Generar nube de palabras'):
            nube = hero.wordcloud(datos['Palabras clave'])
            st.pyplot(nube)
            st.markdown('La nube de palabras permite identificar los temas más recurrentes según las palabras clave de todo el conjunto de datos')
            
        nombres_proyectos = datos['Proyecto'].unique()
        nombre_proyecto = st.sidebar.selectbox(label= 'Selecciona el proyecto', options= nombres_proyectos)
        st.dataframe(data= datos[datos['Proyecto'].str.contains(nombre_proyecto)])
        st.dataframe(data= obtener_recomendacion_ideame(nombre_proyecto))
        
        opcion_uno, opcion_dos = st.beta_columns(2)
        
        with opcion_uno:
            if st.checkbox(label= 'Generar nube de palabras', key= 1):
                nube = hero.wordcloud(obtener_recomendacion_ideame(nombre_proyecto)['Palabras clave'])
                st.pyplot(nube)
                st.markdown('La nube de palabras permite identificar los temas más recurrentes según las palabras clave de los proyectos relacionados conforme a sus métricas y descripciones')
                st.info('Maximiza los gráficos para una mejor visualización')
                
        with opcion_dos:
            if st.checkbox(label= 'Generar gráfico', key= 2):
                plt.figure(figsize= (60, 30))
                fig, ax = plt.subplots()
                ax = sn.scatterplot(data= obtener_recomendacion_ideame(nombre_proyecto), x= 'Recaudación',
                                y= 'Porcentaje', hue= 'Categoría')
                plt.gca().ticklabel_format(axis= 'x', style= 'plain', useOffset= False)
                plt.xticks(rotation= 45)
                plt.title('Relación entre recaudación y porcentaje de compleción de los proyectos relacionados')
                st.pyplot(fig)
                st.info('Maximiza los gráficos para una mejor visualización')
        
if __name__ == '__main__':
    main()



































    
















