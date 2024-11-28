# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Para pruebas estadísticas
# -----------------------------------------------------------------------
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest # para hacer el ztest

def exploracion_dataframe(dataframe, columna_control):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()).head())    
    
    # como estamos en un problema de A/B testing y lo que realmente nos importa es comparar entre el grupo de control y el de test, los principales estadísticos los vamos a sacar de cada una de las categorías
    
    for categoria in dataframe[columna_control].unique():
        dataframe_filtrado = dataframe[dataframe[columna_control] == categoria]
    
        print("\n ..................... \n")
        print(f"Los principales estadísticos de las columnas categóricas para el {categoria.upper()} son: ")
        display(dataframe_filtrado.describe(include = "O").T)
        
        print("\n ..................... \n")
        print(f"Los principales estadísticos de las columnas numéricas para el {categoria.upper()} son: ")
        display(dataframe_filtrado.describe().T)



class Asunciones:
    def __init__(self, dataframe, columna_numerica):

        self.dataframe = dataframe
        self.columna_numerica = columna_numerica
        
    

    def identificar_normalidad_analitica(self, metodo='shapiro', alpha=0.05, verbose=True):
        """
        Evalúa la normalidad de una columna de datos de un DataFrame utilizando la prueba de Shapiro-Wilk o Kolmogorov-Smirnov.

        Params:
            metodo (str): El método a utilizar para la prueba de normalidad ('shapiro' o 'kolmogorov').
            alpha (float): Nivel de significancia para la prueba.
            verbose (bool): Si se establece en True, imprime el resultado de la prueba. Si es False, Returns el resultado.

        Returns:
            bool: True si los datos siguen una distribución normal, False de lo contrario.
        """

        if metodo == 'shapiro':
            _, p_value = stats.shapiro(self.dataframe[self.columna_numerica])
            resultado = p_value > alpha
            mensaje = f"los datos siguen una distribución normal según el test de Shapiro-Wilk." if resultado else f"los datos no siguen una distribución normal según el test de Shapiro-Wilk."
        
        elif metodo == 'kolmogorov':
            _, p_value = stats.kstest(self.dataframe[self.columna_numerica], 'norm')
            resultado = p_value > alpha
            mensaje = f"los datos siguen una distribución normal según el test de Kolmogorov-Smirnov." if resultado else f"los datos no siguen una distribución normal según el test de Kolmogorov-Smirnov."
        else:
            raise ValueError("Método no válido. Por favor, elige 'shapiro' o 'kolmogorov'.")

        if verbose:
            print(f"Para la columna {self.columna_numerica}, {mensaje}")
        else:
            return resultado

        
    def identificar_homogeneidad (self,  columna_categorica):
        
        """
        Evalúa la homogeneidad de las varianzas entre grupos para una métrica específica en un DataFrame dado.

        Params:
        - columna (str): El nombre de la columna que se utilizará para dividir los datos en grupos.
        - columna_categorica (str): El nombre de la columna que se utilizará para evaluar la homogeneidad de las varianzas.

        Returns:
        No Returns nada directamente, pero imprime en la consola si las varianzas son homogéneas o no entre los grupos.
        Se utiliza la prueba de Levene para evaluar la homogeneidad de las varianzas. Si el valor p resultante es mayor que 0.05,
        se concluye que las varianzas son homogéneas; de lo contrario, se concluye que las varianzas no son homogéneas.
        """
        
        # lo primero que tenemos que hacer es crear tantos conjuntos de datos para cada una de las categorías que tenemos, Control Campaign y Test Campaign
        valores_evaluar = []
        
        for valor in self.dataframe[columna_categorica].unique():
            valores_evaluar.append(self.dataframe[self.dataframe[columna_categorica]== valor][self.columna_numerica])

        statistic, p_value = stats.levene(*valores_evaluar)
        if p_value > 0.05:
            print(f"En la variable {columna_categorica} las varianzas son homogéneas entre grupos.")
        else:
            print(f"En la variable {columna_categorica} las varianzas NO son homogéneas entre grupos.")


class Pruebas_parametricas:
    
    def __init__(self, columna_grupo, columna_respuesta, dataframe, categoria_test=None, categoria_control=None):
        """
        Inicializa la clase Pruebas_parametricas.

        Params:
            - columna_grupo: Nombre de la columna que contiene las categorías.
            - columna_respuesta: Nombre de la columna que contiene los datos de respuesta.
            - dataframe: DataFrame que contiene los datos.
            - categoria_test: Valor de la categoría de prueba.
            - categoria_control: Valor de la categoría de control.
        """
        self.columna_grupo = columna_grupo
        self.categoria_test = categoria_test
        self.categoria_control = categoria_control
        self.columna_respuesta = columna_respuesta
        self.dataframe = dataframe

    def separar_grupos_z(self):
        """
        Separa los datos en dos grupos basados en la categoría de prueba y la de control.

        Params: 
            No recibe nigún parámetros
            
        Returns:
            Tupla de dos pandas.Series, uno para cada grupo.
        """
        data_control = self.dataframe[self.dataframe[self.columna_grupo] == self.categoria_control][self.columna_respuesta]
        data_test = self.dataframe[self.dataframe[self.columna_grupo] == self.categoria_test][self.columna_respuesta]

        return data_control, data_test
    
    def separar_grupos(self):
        """
        Genera grupos de datos basados en la columna categórica.
        
        Params: 
            No recibe nigún parámetros

        Returns:
            Una lista de nombres de las categorías.
        """
        lista_categorias =[]
    
        for value in self.dataframe[self.columna_grupo].unique():
            variable_data = self.dataframe[self.dataframe[self.columna_grupo] == value][self.columna_respuesta].values.tolist()
            globals()[value] = variable_data  
            lista_categorias.append(value)
    
        return lista_categorias

    def comprobar_pvalue(self, pvalor, alpha=0.05):
        """
        Comprueba si el valor p es significativo.

        Params:
            - pvalor: Valor p obtenido de la prueba estadística.
            - alpha (opcional): Nivel de significancia. Por defecto es 0.05.

        Returns:
            No devuelve nada.
        """
        if pvalor < alpha:
            print(f"El p-valor de la prueba es {round(pvalor, 2)}, por lo tanto, hay diferencias significativas entre los grupos.")
        else:
            print(f"El p-valor de la prueba es {round(pvalor, 2)}, por lo tanto, no hay evidencia de diferencias significativas entre los grupos.")


    def z_test(self):
        """
        Realiza el test Z para proporciones.

        Calcula el valor Z y el p-valor de la prueba y lo imprime en la consola.

        Params: 
            No recibe nigún parámetros

        Returns:
            No devuelve nada.
        """
        control, test = self.separar_grupos_z()

        # calculamos el número de usuarios que han convertido en cada uno de los tratamientos y creamos una lista (esto es asi porque el método de python para hacer el ztest nos pide una lista)
        convertidos = [control.sum(), test.sum()]
        # contamos el número de filas que tenemos para cada uno de los grupos, es decir, calculamos el tamaño muestral del grupo control y grupo test. Todo esto lo almacenamos en una lista igual que antes. 
        tamaños_muestrales = [control.count(), test.count()]

        resultados_test = proportions_ztest(convertidos, tamaños_muestrales)
        print(f"El estadístico de prueba (Z) es: {round(resultados_test[0], 2)}, el p-valor es {round(resultados_test[1], 2)}")
        
        # Interpretar los resultados
        self.comprobar_pvalue(resultados_test[1])

    def test_anova(self):
        """
        Realiza el test ANOVA para comparar las medias de múltiples grupos.

        Calcula el estadístico F y el valor p de la prueba y lo imprime en la consola.
        
        Params: 
            No recibe nigún parámetros

        Returns:
            No devuelve nada.
        """
        categorias = self.separar_grupos()
        statistic, p_value = stats.f_oneway(*[globals()[var] for var in categorias])

        print("Estadístico F:", statistic)
        print("Valor p:", p_value)

        self.comprobar_pvalue(p_value)

    def test_t(self):
        """
        Realiza el test t de Student para comparar las medias de dos grupos independientes.

        Calcula el estadístico t y el valor p de la prueba y lo imprime en la consola.

        Params: 
            No recibe nigún parámetros

        Returns:
            No devuelve nada.
        """
        categorias = self.separar_grupos()

        t_stat, p_value = stats.ttest_ind(*[globals()[var] for var in categorias])

        print("Estadístico t:", t_stat)
        print("Valor p:", p_value)

        self.comprobar_pvalue(p_value)

    def test_t_dependiente(self):
        """
        Realiza el test t de Student para comparar las medias de dos grupos dependientes.

        Calcula el estadístico t y el valor p de la prueba y lo imprime en la consola.

        Params: 
            No recibe nigún parámetros

        Returns:
            No devuelve nada.
        """
        categorias = self.separar_grupos()

        t_stat, p_value = stats.ttest_rel(*[globals()[var] for var in categorias])

        print("Estadístico t:", t_stat)
        print("Valor p:", p_value)

        self.comprobar_pvalue(p_value)




class Pruebas_no_parametricas:
    def __init__(self, dataframe, variable_respuesta, columna_categorica):
        """
        Inicializa la instancia de la clase TestEstadisticos.

        Parámetros:
        - dataframe: DataFrame de pandas que contiene los datos.
        - variable_respuesta: Nombre de la variable respuesta.
        - columna_categorica: Nombre de la columna que contiene las categorías para comparar.
        """
        self.dataframe = dataframe
        self.variable_respuesta = variable_respuesta
        self.columna_categorica = columna_categorica

    def generar_grupos(self):
        """
        Genera grupos de datos basados en la columna categórica.

        Retorna:
        Una lista de nombres de las categorías.
        """
        lista_categorias =[]
    
        for value in self.dataframe[self.columna_categorica].unique():
            variable_name = value  # Asigna el nombre de la variable
            variable_data = self.dataframe[self.dataframe[self.columna_categorica] == value][self.variable_respuesta].values.tolist()
            globals()[variable_name] = variable_data  
            lista_categorias.append(variable_name)
    
        return lista_categorias

    def comprobar_pvalue(self, pvalor):
        """
        Comprueba si el valor p es significativo.

        Parámetros:
        - pvalor: Valor p obtenido de la prueba estadística.
        """
        if pvalor < 0.05:
            print("Hay una diferencia significativa entre los datos antes y después")
        else:
            print("No hay evidencia suficiente para concluir que hay una diferencia significativa.")

    def test_manwhitneyu(self, categorias): # SE PUEDE USAR SOLO PARA COMPARAR DOS GRUPOS, PERO NO ES NECESARIO QUE TENGAN LA MISMA CANTIDAD DE VALORES
        """
        Realiza el test de Mann-Whitney U.

        Parámetros:
        - categorias: Lista de nombres de las categorías a comparar.
        """
        statistic, p_value = stats.mannwhitneyu(*[globals()[var] for var in categorias])

        print("Estadístico del Test de Mann-Whitney U:", statistic)
        print("Valor p:", p_value)

        self.comprobar_pvalue(p_value)

    def test_wilcoxon(self, categorias): # SOLO LO PODEMOS USAR SI QUEREMOS COMPARAR DOS CATEGORIAS Y SI TIENEN LA MISMA CANTIDAD DE VALORES 
        """
        Realiza el test de Wilcoxon.

        Parámetros:
        - categorias: Lista de nombres de las categorías a comparar.
        """
        statistic, p_value = stats.wilcoxon(*[globals()[var] for var in categorias])

        print("Estadístico del Test de Wilcoxon:", statistic)
        print("Valor p:", p_value)

        # Imprime el estadístico y el valor p
        print("Estadístico de prueba:", statistic)
        print("Valor p:", p_value) 

        self.comprobar_pvalue(p_value)

    def test_kruskal(self, categorias):
       """
       Realiza el test de Kruskal-Wallis.

       Parámetros:
       - categorias: Lista de nombres de las categorías a comparar.
       """
       statistic, p_value = stats.kruskal(*[globals()[var] for var in categorias])

       print("Estadístico de prueba:", statistic)
       print("Valor p:", p_value)

       self.comprobar_pvalue(p_value)