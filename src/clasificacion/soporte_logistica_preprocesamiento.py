# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otros objetivos
# -----------------------------------------------------------------------
import math

# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt


# Para tratar el problema de desbalance
# -----------------------------------------------------------------------
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek


def exploracion_datos(dataframe):

    """
    Realiza una exploración básica de los datos en el DataFrame dado e imprime varias estadísticas descriptivas.

    Parameters:
    -----------
    dataframe : pandas DataFrame. El DataFrame que se va a explorar.

    Returns:
    --------
    None

    Imprime varias estadísticas descriptivas sobre el DataFrame, incluyendo:
    - El número de filas y columnas en el DataFrame.
    - El número de valores duplicados en el DataFrame.
    - Una tabla que muestra las columnas con valores nulos y sus porcentajes.
    - Las principales estadísticas de las variables numéricas en el DataFrame.
    - Las principales estadísticas de las variables categóricas en el DataFrame.

    """

    print(f"El número de filas es {dataframe.shape[0]} y el número de columnas es {dataframe.shape[1]}")

    print("\n----------\n")

    print(f"En este conjunto de datos tenemos {dataframe.duplicated().sum()} valores duplicados")

    
    print("\n----------\n")


    print("Los columnas con valores nulos y sus porcentajes son: ")
    dataframe_nulos = dataframe.isnull().sum()

    display((dataframe_nulos[dataframe_nulos.values >0] / dataframe.shape[0]) * 100)

    print("\n----------\n")
    print("Las principales estadísticas de las variables númericas son:")
    display(dataframe.describe().T)

    print("\n----------\n")
    print("Las principales estadísticas de las variables categóricas son:")
    display(dataframe.describe(include = "O").T)

    print("\n----------\n")
    print("Las características principales del dataframe son:")
    display(dataframe.info())


class Visualizador:
    """
    Clase para visualizar la distribución de variables numéricas y categóricas de un DataFrame.

    Attributes:
    - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.

    Methods:
    - __init__: Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.
    - separar_dataframes: Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.
    - plot_numericas: Grafica la distribución de las variables numéricas del DataFrame.
    - plot_categoricas: Grafica la distribución de las variables categóricas del DataFrame.
    - plot_relacion2: Visualiza la relación entre una variable y todas las demás, incluyendo variables numéricas y categóricas.
    """

    def __init__(self, dataframe):
        """
        Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.

        Parameters:
        - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        """
        self.dataframe = dataframe

    def separar_dataframes(self):
        """
        Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.

        Returns:
        - pandas.DataFrame: DataFrame con variables numéricas.
        - pandas.DataFrame: DataFrame con variables categóricas.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include=["O", "category"])
    
    def plot_numericas(self, color="grey", tamano_grafica=(15, 5)):
        """
        Grafica la distribución de las variables numéricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        lista_num = self.separar_dataframes()[0].columns
        fig, axes = plt.subplots(nrows = 2, ncols = math.ceil(len(lista_num)/2), figsize=tamano_grafica, sharey=True)
        axes = axes.flat
        for indice, columna in enumerate(lista_num):
            sns.histplot(x=columna, data=self.dataframe, ax=axes[indice], color=color, bins=20)
        plt.suptitle("Distribución de variables numéricas")
        plt.tight_layout()

        if len(lista_num) % 2 !=0:
            fig.delaxes(axes[-1])


    def plot_categoricas(self, color="grey", tamano_grafica=(40, 10)):
        """
        Grafica la distribución de las variables categóricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        lista_cat = self.separar_dataframes()[1].columns
        fig, axes = plt.subplots(2, math.ceil(len(lista_cat) / 2), figsize=tamano_grafica)
        axes = axes.flat
        for indice, columna in enumerate(lista_cat):
            sns.countplot(x=columna, data=self.dataframe, order=self.dataframe[columna].value_counts().index,
                          ax=axes[indice], color=color)
            axes[indice].tick_params(rotation=90)
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None)

        plt.suptitle("Distribución de variables categóricas")
        plt.tight_layout()

        if len(lista_cat) % 2 !=0:
            fig.delaxes(axes[-1])


    def plot_relacion(self, vr, tamano_grafica=(40, 12)):


        lista_num = self.separar_dataframes()[0].columns
        lista_cat = self.separar_dataframes()[1].columns

        fig, axes = plt.subplots(3, int(len(self.dataframe.columns) / 3), figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.dataframe.columns):
            if columna == vr:
                fig.delaxes(axes[indice])
            elif columna in lista_num:
                sns.histplot(x = columna, 
                             hue = vr, 
                             data = self.dataframe, 
                             ax = axes[indice], 
                             palette = "magma", 
                             legend = False)
                
            elif columna in lista_cat:
                sns.countplot(x = columna, 
                              hue = vr, 
                              data = self.dataframe, 
                              ax = axes[indice], 
                              palette = "magma"
                              )

            axes[indice].set_title(f"Relación {columna} vs {vr}")   

        plt.tight_layout()
    
        
    def deteccion_outliers(self, color = "grey"):

        """
        Detecta y visualiza valores atípicos en un DataFrame.

        Params:
            - dataframe (pandas.DataFrame):  El DataFrame que se va a usar

        Returns:
            No devuelve nada

        Esta función selecciona las columnas numéricas del DataFrame dado y crea un diagrama de caja para cada una de ellas para visualizar los valores atípicos.
        """

        lista_num = self.separar_dataframes()[0].columns

        fig, axes = plt.subplots(2, ncols = math.ceil(len(lista_num)/2), figsize=(15,5))
        axes = axes.flat

        for indice, columna in enumerate(lista_num):
            sns.boxplot(x=columna, data=self.dataframe, 
                        ax=axes[indice], 
                        color=color, 
                        flierprops={'markersize': 4, 'markerfacecolor': 'orange'})

        if len(lista_num) % 2 != 0:
            fig.delaxes(axes[-1])

        
        plt.tight_layout()

    def correlacion(self, tamano_grafica = (7, 5)):

        """
        Visualiza la matriz de correlación de un DataFrame utilizando un mapa de calor.

        Params:
            - dataframe : pandas DataFrame. El DataFrame que contiene los datos para calcular la correlación.

        Returns:
        No devuelve nada 

        Muestra un mapa de calor de la matriz de correlación.

        - Utiliza la función `heatmap` de Seaborn para visualizar la matriz de correlación.
        - La matriz de correlación se calcula solo para las variables numéricas del DataFrame.
        - La mitad inferior del mapa de calor está oculta para una mejor visualización.
        - Permite guardar la imagen del mapa de calor como un archivo .png si se solicita.

        """

        plt.figure(figsize = tamano_grafica )

        mask = np.triu(np.ones_like(self.dataframe.corr(numeric_only=True), dtype = np.bool_))

        sns.heatmap(data = self.dataframe.corr(numeric_only = True), 
                    annot = True, 
                    vmin=-1,
                    vmax=1,
                    cmap="magma",
                    linecolor="black", 
                    fmt='.1g', 
                    mask = mask)
    

class Desbalanceo:
    def __init__(self, dataframe, variable_dependiente):
        self.dataframe = dataframe
        self.variable_dependiente = variable_dependiente

    def visualizar_clase(self, color="orange", edgecolor="black"):
        plt.figure(figsize=(8, 5))  # para cambiar el tamaño de la figura
        fig = sns.countplot(data=self.dataframe, 
                            x=self.variable_dependiente,  
                            color=color,  
                            edgecolor=edgecolor)
        fig.set(xticklabels=["No", "Yes"])
        plt.show()

    def balancear_clases_pandas(self, metodo):

        # Contar las muestras por clase
        contar_clases = self.dataframe[self.variable_dependiente].value_counts()
        clase_mayoritaria = contar_clases.idxmax()
        clase_minoritaria = contar_clases.idxmin()

        # Separar las clases
        df_mayoritaria = self.dataframe[self.dataframe[self.variable_dependiente] == clase_mayoritaria]
        df_minoritaria = self.dataframe[self.dataframe[self.variable_dependiente] == clase_minoritaria]

        if metodo == "downsampling":
            # Submuestrear la clase mayoritaria
            df_majority_downsampled = df_mayoritaria.sample(contar_clases[clase_minoritaria], random_state=42)
            # Combinar los subconjuntos
            df_balanced = pd.concat([df_majority_downsampled, df_minoritaria])

        elif metodo == "upsampling":
            # Sobremuestrear la clase minoritaria
            df_minority_upsampled = df_minoritaria.sample(contar_clases[clase_mayoritaria], replace=True, random_state=42)
            # Combinar los subconjuntos
            df_balanced = pd.concat([df_mayoritaria, df_minority_upsampled])

        else:
            raise ValueError("Método no reconocido. Use 'downsampling' o 'upsampling'.")

        return df_balanced

    def balancear_clases_imblearn(self, metodo):

        X = self.dataframe.drop(columns=[self.variable_dependiente])
        y = self.dataframe[self.variable_dependiente]

        if metodo == "RandomOverSampler":
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)

        elif metodo == "RandomUnderSampler":
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)

        else:
            raise ValueError("Método no reconocido. Use 'RandomOverSampler' o 'RandomUnderSampler'.")

        df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name=self.variable_dependiente)], axis=1)
        return df_resampled
    
    def balancear_clases_smote(self):
        X = self.dataframe.drop(columns=[self.variable_dependiente])
        y = self.dataframe[self.variable_dependiente]

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=self.variable_dependiente)], axis=1)
        return df_resampled

    def balancear_clases_smote_tomek(self):
        X = self.dataframe.drop(columns=[self.variable_dependiente])
        y = self.dataframe[self.variable_dependiente]

        smote_tomek = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=self.variable_dependiente)], axis=1)
        return df_resampled

