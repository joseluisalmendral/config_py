# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otros objetivos
# -----------------------------------------------------------------------
import math
from scipy.stats import chi2_contingency


# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuraciones
# -----------------------------------------------------------------------
pd.set_option('display.max_info_columns', 50)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


#! FUNCIONES
#!-------------

def exploracion_datos(dataframe: pd.DataFrame, nunique=True, info=False):

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

    print("Las principales estadísticas de las variables númericas son:")
    display(dataframe.describe().T)

    print("\n----------\n")
    print("Las principales estadísticas de las variables categóricas son:")
    display(dataframe.describe(include = "O").T)


    print("\n----------\n")
    print("Los columnas con valores nulos y sus porcentajes son: ")
    dataframe_nulos = dataframe.isnull().sum()
    display(pd.DataFrame((dataframe_nulos[dataframe_nulos.values >0] / dataframe.shape[0]) * 100, columns=['%_nulos']).sort_values(by='%_nulos', ascending=False))

    if nunique:
        print("\n----------\n")
        print("Valores unicos que las variables")
        display(dataframe.nunique().reset_index())

    if info: 
        print("\n----------\n")
        print("Las características principales del dataframe son:")
        display(dataframe.info())




#! CLASE
#!-------------

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
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include="O")
    
    def plot_numericas(self, color="grey", tamano_grafica=(15, 5), kde = False):
        """
        Grafica la distribución de las variables numéricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        lista_num = self.separar_dataframes()[0].columns
        _, axes = plt.subplots(nrows = 2, ncols = math.ceil(len(lista_num)/2), figsize=tamano_grafica, sharey=True)
        axes = axes.flat
        for indice, columna in tqdm(enumerate(lista_num)):
            sns.histplot(x=columna, data=self.dataframe, ax=axes[indice], color=color, bins=20, kde=kde)
        plt.suptitle("Distribución de variables numéricas")
        plt.tight_layout()

    def plot_categoricas(self, color="grey", tamano_grafica=(40, 10)):
        """
        Grafica la distribución de las variables categóricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        dataframe_cat = self.separar_dataframes()[1]
        _, axes = plt.subplots(2, math.ceil(len(dataframe_cat.columns) / 2), figsize=tamano_grafica)
        axes = axes.flat
        for indice, columna in enumerate(dataframe_cat.columns):
            sns.countplot(x=columna, data=self.dataframe, order=self.dataframe[columna].value_counts().index,
                          ax=axes[indice], color=color)
            axes[indice].tick_params(rotation=90)
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None)

        plt.tight_layout()
        plt.suptitle("Distribución de variables categóricas")


    def visualizar_categoricas_numericas(self):
        """
        Genera gráficos de dispersión para las variables numéricas vs todas las variables categóricas.

        Params:
            - Ninguno.

        Returns:
            - None.
        """
        categorical_columns = self.dataframe.select_dtypes(include=['object', 'category']).columns
        numerical_columns = self.dataframe.select_dtypes(include=np.number).columns
        if len(categorical_columns) > 0:
            for num_col in numerical_columns:
                try:
                    _, axes = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(10, 9 * len(categorical_columns)))
                    axes = axes.flat
                    plt.suptitle(f'Dispersión {num_col} vs variables categóricas', fontsize=24)
                    for indice, cat_col in enumerate(categorical_columns):
                        sns.scatterplot(x=num_col, y=self.dataframe.index, hue=cat_col, data=self.dataframe, ax=axes[indice])
                        axes[indice].set_xlabel(num_col, fontsize=16)
                        axes[indice].set_ylabel('Índice', fontsize=16)
                        axes[indice].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
                    plt.tight_layout()
                except: 
                    sns.scatterplot(x=num_col, y=self.dataframe.index, hue=categorical_columns[0], data=self.dataframe)
                    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=10)
                    plt.xlabel(num_col, fontsize=16)
                    plt.ylabel('Índice', fontsize=16)
                    plt.tight_layout()
        else:
            print("No hay columnas categóricas en el DataFrame.")


    def plot_relacion(self, vr, tamano_grafica=(20, 10)):


        lista_num = self.separar_dataframes()[0].columns
        lista_cat = self.separar_dataframes()[1].columns

        fig, axes = plt.subplots(ncols = 2, nrows = math.ceil(len(self.dataframe.columns) / 2), figsize=tamano_grafica)
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

            axes[indice].set_title(f"Relación {columna} vs {vr}",size=25)   

        plt.tight_layout()
    
    def analisis_temporal(self, var_respuesta, var_temporal, color = "black", order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):


        """
        Realiza un análisis temporal mensual de una variable de respuesta en relación con una variable temporal. Visualiza un gráfico de líneas que muestra la relación entre la variable de respuesta y la variable temporal (mes), con la línea de la media de la variable de respuesta.


        Params:
        -----------
        dataframe : pandas DataFrame. El DataFrame que contiene los datos.
        var_respuesta : str. El nombre de la columna que contiene la variable de respuesta.
        var_temporal : str. El nombre de la columna que contiene la variable temporal (normalmente el mes).
        order : list, opcional.  El orden de los meses para representar gráficamente. Por defecto, se utiliza el orden estándar de los meses.

        Returns:
        --------
        None

 
        """


        plt.figure(figsize = (15, 5))

        # Convierte la columna "Month" en un tipo de datos categórico con el orden especificado
        self.dataframe[var_temporal] = pd.Categorical(self.dataframe[var_temporal], categories=order, ordered=True)

        # Trama el gráfico
        sns.lineplot(x=var_temporal, 
                     y=var_respuesta, 
                     data=self.dataframe, 
                     color = color)

        # Calcula la media de PageValues
        mean_page_values = self.dataframe[var_respuesta].mean()

        # Agrega la línea de la media
        plt.axhline(mean_page_values, 
                    color='green', 
                    linestyle='--', 
                    label='Media de PageValues')


        # quita los ejes de arriba y de la derecha
        sns.despine()

        # Rotula el eje x
        plt.xlabel("Month");


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
                    cmap="viridis",
                    linecolor="black", 
                    fmt='.2g', 
                    mask = mask)
    


def detectar_orden_cat_bina(df, lista_cat, var_respuesta, sig_level = 0.05):
    for categoria in lista_cat:
        print(f"Estamos evaluando la variable {categoria.upper()}")
        df_crosstab = pd.crosstab(df[categoria], df[var_respuesta])
        display(df_crosstab)
        chi2, p, dof, expected = chi2_contingency(df_crosstab)

        if p<sig_level:
            print(f"Para la categoría {categoria.upper()} SÍ hay diferencias significativas, p = {p:.4f}")
            display(pd.DataFrame(expected, index = df_crosstab.index, columns = df_crosstab.columns).round())
        else:
            print(f"Para la categoría {categoria.upper()} NO hay diferencias significativas, p = {p:.4f}\n")
        print("--------"*10)