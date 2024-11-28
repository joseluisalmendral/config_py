
# Visualización
# ------------------------------------------------------------------------------
import seaborn as sns 
import matplotlib.pyplot as plt

# Otras librerias
# ------------------------------------------------------------------------------
import math

# Para crear combinaciones de columnas
# ------------------------------------------------------------------------------
from itertools import combinations

# Para hacer tablas de contingencia
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np


def identificar_linealidad(dataframe, lista_combinacion_columnas):
    """
    Visualiza la relación lineal entre pares de columnas numéricas especificadas en el DataFrame mediante gráficos de dispersión.

    Params
        - dataframe : pandas.DataFrame. El DataFrame que contiene los datos.
        - lista_combinacion_columnas : list of tuple. Una lista de tuplas, donde cada tupla contiene dos nombres de columnas numéricas cuyas relaciones se desean visualizar.

    Returns
        La función genera una visualización de los gráficos de dispersión y no devuelve ningún valor.
    """
    num_filas = math.ceil(len(lista_combinacion_columnas) / 2)

    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=(19, 11))
    axes = axes.flat

    for indice, columnas in enumerate(lista_combinacion_columnas):
        sns.scatterplot(x=columnas[0], y=columnas[1], data=dataframe, ax=axes[indice])
        axes[indice].set_title(f"Relación entre {columnas[0]} y {columnas[1]}")

    if len(lista_combinacion_columnas) % 2 != 0:
        fig.delaxes(axes[-1])

    fig.suptitle("Relación Entre Variables Numéricas")
    plt.tight_layout()
    plt.show()


def visualizar_tablas_frecuencias(dataframe, lista_categorias):
    """
    Visualiza las tablas de frecuencias para las columnas categóricas especificadas en el DataFrame.

    Params
        - dataframe : pandas.DataFrame.El DataFrame que contiene los datos.
        - lista_categorias : list of str. Una lista de nombres de columnas categóricas para las cuales se desean visualizar las tablas de frecuencias.

    Returns
        La función genera una visualización de las tablas de frecuencias y no devuelve ningún valor.
    """
    num_filas = math.ceil(len(lista_categorias) / 2)

    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=(19, 11))
    axes = axes.flat

    for indice, columna in enumerate(lista_categorias):
        sns.countplot(x=columna, data=dataframe, ax=axes[indice])
        axes[indice].set_title(f"Distribución de la columna {columna}")
        axes[indice].set_xlabel("")

    if len(lista_categorias) % 2 != 0:
        fig.delaxes(axes[-1])

    fig.suptitle("Distribución Variables Categóricas")
    plt.tight_layout()
    plt.show()


def visualizar_tablas_contingencia(dataframe, lista_col_categorias):
    """
    Visualiza tablas de contingencia para todas las combinaciones posibles de las variables categóricas especificadas en el DataFrame.

    Params
        - dataframe : pandas.DataFrame. El DataFrame que contiene los datos.
        - lista_col_categorias : list of str. Una lista de nombres de columnas categóricas para las cuales se desean visualizar las tablas de contingencia.

    Returns
        La función genera una visualización de las tablas de contingencia y no devuelve ningún valor.
    """
    # Crear todas las posibles combinaciones de variables categóricas
    combinaciones_categoricas = list(combinations(lista_col_categorias, 2))

    num_filas = math.ceil(len(combinaciones_categoricas) / 2)

    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=(40, 30))
    axes = axes.flat

    # Generar las tablas de contingencia para cada relación de variables
    for indice, columnas in enumerate(combinaciones_categoricas):
        tabla_contingencia = pd.crosstab(dataframe[columnas[0]], dataframe[columnas[1]])
        sns.heatmap(tabla_contingencia, 
                    annot=True, 
                    cmap="YlGnBu",
                    ax=axes[indice])
        axes[indice].set_title(f"Tabla de contingencia {columnas[0]} y {columnas[1]}")

    plt.suptitle("Tablas contingencias del DataFrame")
    if len(combinaciones_categoricas) % 2 != 0:
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()



def visualizar_medidas_posicion(dataframe, columna, percentiles=[10, 25, 50, 75, 90]):
    """
    Visualiza un histograma de la columna especificada del DataFrame junto con líneas que representan los percentiles dados.

    Params
    
        - dataframe : pandas.DataFrame. El DataFrame que contiene los datos.
    
        - columna : str. El nombre de la columna para la cual se quiere visualizar el histograma y los percentiles.
    
        - percentiles : list of int, optional. Una lista de percentiles a calcular y mostrar en el histograma. El valor por defecto es [10, 25, 50, 75, 90].

    Returns
        La función genera una visualización y no devuelve ningún valor.
    """
    valores_percentiles = np.percentile(dataframe[columna], percentiles)

    sns.histplot(x=columna, 
                 data=dataframe,
                 bins=30, 
                 edgecolor="black", 
                 color="orange")
   
    # Añadir líneas de percentiles
    for percentile, value in zip(percentiles, valores_percentiles):
        plt.axvline(value, color='green', linestyle='--', label=f'{percentile} percentil')
    
    plt.title('Histograma con Percentiles')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.legend()

