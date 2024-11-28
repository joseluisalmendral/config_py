# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

import math 
# Imputación de nulos usando métodos avanzados estadísticos
# -----------------------------------------------------------------------
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

# Visualización de datos
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns



#! FUNCIONES
#!-------------

def porcentaje_nulos(df):
    """
    Calcula el porcentaje de valores nulos por columna en un DataFrame.

    Esta función evalúa todas las columnas del DataFrame y devuelve un 
    Series con los porcentajes de valores nulos de las columnas que 
    tienen al menos un valor nulo, ordenados en orden descendente.

    Parameters:
    ----------
    df : pd.DataFrame
        El DataFrame a analizar para calcular el porcentaje de valores nulos.

    Returns:
    -------
    pd.Series
        Un Series donde el índice son los nombres de las columnas con 
        valores nulos, y los valores son los porcentajes de nulos, 
        ordenados en orden descendente.
    
    Notes:
    -----
    - Las columnas sin valores nulos no se incluyen en el resultado.
    - El porcentaje de valores nulos se calcula como el promedio de valores
      nulos en cada columna multiplicado por 100.
    """
    porcentaje = df.isnull().mean() * 100
    return porcentaje[porcentaje > 0].sort_values(ascending=False)


def obtener_pesos_variables(serie: pd.Series):
    """
    Calcula los valores únicos de una serie y su distribución porcentual.

    Esta función toma una serie de pandas, calcula los valores únicos y 
    la proporción de cada valor en relación con el total de la serie.
    Los porcentajes se redondean a dos decimales.

    Parameters:
    ----------
    serie : pd.Series
        Serie de pandas de la cual se calcularán los valores únicos y 
        sus porcentajes.

    Returns:
    -------
    valores : pd.Index
        Índices de los valores únicos presentes en la serie.
    porcentajes : np.ndarray
        Arreglo de los porcentajes normalizados de cada valor único, 
        redondeados a dos decimales.
    """
    serie_values_normalize = serie.value_counts(normalize=True)
    valores = serie_values_normalize.index
    porcentajes = np.round(serie_values_normalize.values, 2)

    # Ajustar para que la suma sea exactamente 1
    porcentajes = porcentajes / porcentajes.sum()
    
    return valores, porcentajes


def imputar_mediante_pesos(dataframe: pd.DataFrame, columnas: list):
    """
    Imputa valores nulos en columnas seleccionadas de un DataFrame basándose en 
    la distribución de frecuencias de los valores existentes en cada columna.

    Para cada columna especificada, los valores nulos se reemplazan por valores 
    aleatorios seleccionados de la distribución de frecuencias calculada de 
    los datos no nulos de la columna. Las probabilidades para la imputación 
    corresponden a las proporciones relativas de cada valor único.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        El DataFrame que contiene las columnas con valores nulos a imputar.
    columnas : list
        Lista de nombres de columnas del DataFrame en las que se realizará 
        la imputación de valores nulos.

    Returns:
    -------
    None
        La función modifica el DataFrame original in-place, reemplazando 
        los valores nulos en las columnas especificadas.

    Notes:
    -----
    - La función utiliza la función `obtener_pesos_variables` para calcular los 
      valores únicos y las probabilidades de cada columna.
    - La imputación utiliza `np.random.choice`, lo que implica que los valores 
      imputados serán diferentes en cada ejecución, dado que la selección es aleatoria.
    """
    for columna in columnas:
        valores, probabilidades = obtener_pesos_variables(dataframe[columna])
        filtro = dataframe[columna].isnull()
        
        dataframe.loc[filtro, columna] = np.random.choice(valores, size=filtro.sum(), p=probabilidades)


def sustituir_valores_df(df_original: pd.DataFrame, df_nuevo: pd.DataFrame, columnas):
    """
    Sustituye los valores de columnas específicas en un DataFrame original 
    por los valores correspondientes de otro DataFrame.

    Para cada columna especificada, los valores en el DataFrame original 
    son reemplazados directamente por los valores de la misma columna 
    en el DataFrame nuevo. Es importante que ambos DataFrames tengan los 
    mismos índices para que los valores se alineen correctamente.

    Parameters:
    ----------
    df_original : pd.DataFrame
        El DataFrame cuyos valores de columna serán reemplazados.
    df_nuevo : pd.DataFrame
        El DataFrame que contiene los nuevos valores para sustituir.
    columnas : list
        Lista de nombres de las columnas que serán reemplazadas.

    Returns:
    -------
    None
        La función modifica el DataFrame original in-place, actualizando 
        las columnas especificadas.

    Notes:
    -----
    - Ambas columnas en los DataFrames deben estar alineadas por índice. 
      Si no lo están, los resultados pueden ser inconsistentes.
    - Si alguna de las columnas en `columnas` no existe en `df_nuevo`, 
      se generará un error KeyError.
    """
    for columna in columnas:
        df_original[columna] = df_nuevo[columna]


#! CLASE
#!-------------

class GestionNulos:
    """
    Clase para gestionar los valores nulos en un DataFrame.
    """

    def __init__(self, dataframe):
        """
        Inicializa la clase con un DataFrame.

        Parámetros:
        - dataframe: DataFrame de pandas.
        """
        self.dataframe = dataframe
    
    def calcular_porcentaje_nulos(self):
        """
        Calcula el porcentaje de valores nulos en el DataFrame.

        Retorna:
        - Series: Porcentaje de valores nulos para cada columna con valores nulos.
        """
        df_nulos = (self.dataframe.isnull().sum() / self.dataframe.shape[0]) * 100
        return df_nulos[df_nulos > 0]
    
    def seleccionar_columnas_nulas(self):
        """
        Selecciona las columnas con valores nulos.

        Retorna:
        - Tuple: Tupla de dos elementos con las columnas categóricas y numéricas que tienen valores nulos.
        """
        nulos_esta_cat = self.dataframe[self.dataframe.columns[self.dataframe.isnull().any()]].select_dtypes(include="O").columns
        nulos_esta_num = self.dataframe[self.dataframe.columns[self.dataframe.isnull().any()]].select_dtypes(include=np.number).columns
        return nulos_esta_cat, nulos_esta_num

    def mostrar_distribucion_categoricas(self):
        """
        Muestra la distribución de categorías para las columnas categóricas con valores nulos.
        """
        col_categoricas = self.seleccionar_columnas_nulas()[0]
        for col in col_categoricas:
            print(f"La distribución de las categorías para la columna {col.upper()}")
            display(self.dataframe[col].value_counts(normalize=True))
            print("........................")

    def imputar_nulos_categoricas(self, lista_moda, lista_nueva_cat):
        """
        Imputa los valores nulos en las columnas categóricas.

        Parámetros:
        - lista_moda: Lista de nombres de columnas donde se imputarán los valores nulos con la moda.
        - lista_nueva_cat: Lista de nombres de columnas donde se imputarán los valores nulos con una nueva categoría "Unknown".

        Retorna:
        - DataFrame: DataFrame con los valores nulos imputados.
        """
        # Imputar valores nulos con moda
        moda_diccionario = {col: self.dataframe[col].mode()[0] for col in lista_moda}
        self.dataframe.fillna(moda_diccionario, inplace=True)

        # Imputar valores nulos con "Unknown"
        self.dataframe[lista_nueva_cat] = self.dataframe[lista_nueva_cat].fillna("Unknown")
    
        return self.dataframe
    
    def identificar_nulos_numericas(self, tamano_grafica=(20, 15)):
        """
        Identifica y visualiza valores nulos en las columnas numéricas mediante gráficos de caja.

        Parámetros:
        - tamano_grafica: Tamaño de las gráficas de caja.
        """
        col_numericas = self.seleccionar_columnas_nulas()[1]

        num_cols = len(col_numericas)
        num_filas = math.ceil(num_cols / 2)

        fig, axes = plt.subplots(num_filas, 2, figsize=tamano_grafica)
        axes = axes.flat

        for indice, col in enumerate(col_numericas):
            sns.boxplot(x=col, data=self.dataframe, ax=axes[indice])
            
        plt.tight_layout();

    def imputar_knn(self, lista_columnas_knn):
        """
        Imputa los valores nulos en las columnas numéricas utilizando el algoritmo KNN.

        Parámetros:
        - lista_columnas_knn: Lista de nombres de columnas numéricas donde se imputarán los valores nulos.

        Retorna:
        - DataFrame: DataFrame con los valores nulos imputados.
        """
        imputer_knn = KNNImputer(n_neighbors=5)
        knn_imputado = imputer_knn.fit_transform(self.dataframe[lista_columnas_knn])

        nuevas_columnas_knn = [col + "_knn" for col in lista_columnas_knn]
        self.dataframe[nuevas_columnas_knn] = knn_imputado

        return self.dataframe
    
    def imputar_imputer(self, lista_columnas_iterative):
        """
        Imputa los valores nulos en las columnas numéricas utilizando el método IterativeImputer.

        Parámetros:
        - lista_columnas_iterative: Lista de nombres de columnas numéricas donde se imputarán los valores nulos.

        Retorna:
        - DataFrame: DataFrame con los valores nulos imputados.
        """
        imputer_iterative = IterativeImputer(max_iter=20, random_state=42)
        iterative_imputado = imputer_iterative.fit_transform(self.dataframe[lista_columnas_iterative])

        nuevas_columnas_iter = [col + "_iterative" for col in lista_columnas_iterative]
        self.dataframe[nuevas_columnas_iter] = iterative_imputado

        return self.dataframe
    
    def comparar_metodos(self):
        """
        Compara los resultados de imputación de los métodos KNN y IterativeImputer.
        """
        columnas_seleccionadas = self.dataframe.columns[self.dataframe.columns.str.contains("_knn|_iterative")].tolist() + self.seleccionar_columnas_nulas()[1].tolist()
        resultados = self.dataframe.describe()[columnas_seleccionadas].reindex(sorted(columnas_seleccionadas), axis=1)
        return resultados

    def columnas_eliminar(self, lista_columnas_eliminar):
        return self.dataframe.drop(lista_columnas_eliminar, axis = 1, inplace = True)