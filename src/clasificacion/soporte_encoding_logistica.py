# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Para la visualización 
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Otros objetivos
# -----------------------------------------------------------------------
import math
from itertools import combinations


# Para pruebas estadísticas
# -----------------------------------------------------------------------
from scipy.stats import chi2_contingency

# Para la codificación de las variables numéricas
# -----------------------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder # para poder aplicar los métodos de OneHot, Ordinal,  Label y Target Encoder 
from category_encoders import TargetEncoder
import pickle

class AnalisisChiCuadrado:
    def __init__(self, dataframe, variable_predictora, variable_respuesta):
        """
        Inicializa la clase con el DataFrame y las dos columnas a analizar.

        Parámetros:
        - dataframe: DataFrame de pandas que contiene los datos.
        - columna1: Nombre de la primera columna (por ejemplo, la variable dependiente).
        - columna2: Nombre de la segunda columna (por ejemplo, la variable categórica).
        """
        self.dataframe = dataframe
        self.variable_predictora = variable_predictora
        self.variable_respuesta = variable_respuesta
        self.tabla_contingencia = None
        self.resultado_chi2 = None

    def generar_tabla_contingencia(self):
        """
        Genera una tabla de contingencia para las dos columnas especificadas.

        Retorna:
        - pandas.DataFrame: La tabla de contingencia.
        """
        self.tabla_contingencia = pd.crosstab(self.dataframe[self.variable_respuesta], self.dataframe[self.variable_predictora])
        print("Tabla de contingencia:")
        display(self.tabla_contingencia)
        return self.tabla_contingencia

    def realizar_prueba_chi_cuadrado(self):
        """
        Realiza la prueba de Chi-cuadrado para la tabla de contingencia generada.

        Retorna:
        - dict: Un diccionario con los resultados de la prueba (chi2, p-valor, grados de libertad, tabla esperada).
        """
        if self.tabla_contingencia is None:
            raise ValueError("Primero debes generar la tabla de contingencia utilizando 'generar_tabla_contingencia'.")

        chi2, p, dof, expected = chi2_contingency(self.tabla_contingencia)
        self.resultado_chi2 = {
            "Chi2": chi2,
            "p_valor": p
        }

        print(f"\nResultado de la prueba de Chi-cuadrado:")
        print(f"Chi2: {chi2}, p-valor: {p}")
        
        # Interpretación del p-valor
        if p < 0.05:
            print("El p-valor < 0.05, parece que hay diferencias entre los grupos.")
        else:
            print("El p-valor >= 0.05, no hay diferencias entre los grupos.")


class Encoding:
    """
    Clase para realizar diferentes tipos de codificación en un DataFrame.

    Parámetros:
        - dataframe: DataFrame de pandas, el conjunto de datos a codificar.
        - diccionario_encoding: dict, un diccionario que especifica los tipos de codificación a realizar.
        - variable_respuesta: str, el nombre de la variable objetivo.

    Métodos:
        - one_hot_encoding(): Realiza codificación one-hot en las columnas especificadas en el diccionario de codificación.
        - get_dummies(prefix='category', prefix_sep='_'): Realiza codificación get_dummies en las columnas especificadas en el diccionario de codificación.
        - ordinal_encoding(): Realiza codificación ordinal en las columnas especificadas en el diccionario de codificación.
        - label_encoding(): Realiza codificación label en las columnas especificadas en el diccionario de codificación.
        - target_encoding(): Realiza codificación target en la variable especificada en el diccionario de codificación.
        - frequency_encoding(): Realiza codificación de frecuencia en las columnas especificadas en el diccionario de codificación.
    """

    def __init__(self, dataframe, variable_respuesta, diccionario_encoding):
        self.dataframe = dataframe
        self.diccionario_encoding = diccionario_encoding
        self.variable_respuesta = variable_respuesta
    
    def one_hot_encoding(self):
        """
        Realiza codificación one-hot en las columnas especificadas en el diccionario de codificación.

        Returns:
            - dataframe: DataFrame de pandas, el DataFrame con codificación one-hot aplicada.
        """
        # accedemos a la clave de 'onehot' para poder extraer las columnas a las que que queramos aplicar OneHot Encoding. En caso de que no exista la clave, esta variable será una lista vacía
        col_encode = self.diccionario_encoding.get("onehot", [])

        # si hay contenido en la lista 
        if col_encode:

            # instanciamos la clase de OneHot
            one_hot_encoder = OneHotEncoder()

            # transformamos los datos de las columnas almacenadas en la variable col_code
            trans_one_hot = one_hot_encoder.fit_transform(self.dataframe[col_encode])

            # el objeto de la transformación del OneHot es necesario convertilo a array (con el método toarray()), para luego convertilo a DataFrame
            # además, asignamos el valor de las columnas usando el método get_feature_names_out()
            oh_df = pd.DataFrame(trans_one_hot.toarray(), columns=one_hot_encoder.get_feature_names_out())

            # concatenamos los resultados obtenidos en la transformación con el DataFrame original
            self.dataframe = pd.concat([self.dataframe.reset_index(drop=True), oh_df.reset_index(drop=True)], axis=1)
        
        self.dataframe.drop(columns=col_encode, inplace=True)
        # Guardar el modelo
        with open('transformer_one.pkl', 'wb') as f:
            pickle.dump(one_hot_encoder, f)
        return self.dataframe
    
    def get_dummies(self, prefix='category', prefix_sep="_"):
        """
        Realiza codificación get_dummies en las columnas especificadas en el diccionario de codificación.

        Parámetros:
        - prefix: str, prefijo para los nombres de las nuevas columnas codificadas.
        - prefix_sep: str, separador entre el prefijo y el nombre original de la columna.

        Returns:
        - dataframe: DataFrame de pandas, el DataFrame con codificación get_dummies aplicada.
        """
        # accedemos a la clave de 'dummies' para poder extraer las columnas a las que que queramos aplicar este método. En caso de que no exista la clave, esta variable será una lista vacía
        col_encode = self.diccionario_encoding.get("dummies", [])

        if col_encode:
            # aplicamos el método get_dummies a todas las columnas seleccionadas, y determinamos su prefijo y la separación
            df_dummies = pd.get_dummies(self.dataframe[col_encode], dtype=int, prefix=prefix, prefix_sep=prefix_sep)
            
            # concatenamos los resultados del get_dummies con el DataFrame original
            self.dataframe = pd.concat([self.dataframe.reset_index(drop=True), df_dummies.reset_index(drop=True)], axis=1)
            
            # eliminamos las columnas original que ya no nos hacen falta
            self.dataframe.drop(col_encode, axis=1, inplace=True)
    
        return self.dataframe

    def ordinal_encoding(self):
        """
        Realiza codificación ordinal en las columnas especificadas en el diccionario de codificación.

        Returns:
        - dataframe: DataFrame de pandas, el DataFrame con codificación ordinal aplicada.
        """

        # Obtenemos las columnas a codificar bajo la clave 'ordinal'. Si no existe la clave, la variable col_encode será una lista vacía.
        col_encode = self.diccionario_encoding.get("ordinal", {})

        # Verificamos si hay columnas a codificar.
        if col_encode:

            # Obtenemos las categorías de cada columna especificada para la codificación ordinal.
            orden_categorias = list(self.diccionario_encoding["ordinal"].values())
            
            # Inicializamos el codificador ordinal con las categorías especificadas.
            ordinal_encoder = OrdinalEncoder(categories=orden_categorias, dtype=float, handle_unknown="use_encoded_value", unknown_value=np.nan)
            
            # Aplicamos la codificación ordinal a las columnas seleccionadas.
            ordinal_encoder_trans = ordinal_encoder.fit_transform(self.dataframe[col_encode.keys()])

            # Eliminamos las columnas originales del DataFrame.
            self.dataframe.drop(col_encode, axis=1, inplace=True)
            
            # Creamos un nuevo DataFrame con las columnas codificadas y sus nombres.
            ordinal_encoder_df = pd.DataFrame(ordinal_encoder_trans, columns=ordinal_encoder.get_feature_names_out())

            # Concatenamos el DataFrame original con el DataFrame de las columnas codificadas.
            self.dataframe = pd.concat([self.dataframe.reset_index(drop=True), ordinal_encoder_df], axis=1)

        return self.dataframe


    def label_encoding(self):
        """
        Realiza codificación label en las columnas especificadas en el diccionario de codificación.

        Returns:
        - dataframe: DataFrame de pandas, el DataFrame con codificación label aplicada.
        """

        # accedemos a la clave de 'label' para poder extraer las columnas a las que que queramos aplicar Label Encoding. En caso de que no exista la clave, esta variable será una lista vacía
        col_encode = self.diccionario_encoding.get("label", [])

        # si hay contenido en la lista 
        if col_encode:

            # instanciamos la clase LabelEncoder()
            label_encoder = LabelEncoder()

            # aplicamos el Label Encoder a cada una de las columnas, y creamos una columna con el nombre de la columna (la sobreescribimos)
            self.dataframe[col_encode] = self.dataframe[col_encode].apply(lambda col: label_encoder.fit_transform(col))
     
        return self.dataframe

    def target_encoding(self):
        """
        Realiza codificación target en la variable especificada en el diccionario de codificación.

        Returns:
        - dataframe: DataFrame de pandas, el DataFrame con codificación target aplicada.
        """
        
        # accedemos a la clave de 'target' para poder extraer las columnas a las que que queramos aplicar Target Encoding. En caso de que no exista la clave, esta variable será una lista vacía
        col_encode = self.diccionario_encoding.get("target", [])

        # si hay contenido en la lista 
        if col_encode:
            print(col_encode)
  
            target_encoder = TargetEncoder(cols=col_encode)
            variables_encoded = target_encoder.fit_transform(self.dataframe, self.dataframe[self.variable_respuesta])
        # Guardar el modelo
        with open('transformer_target.pkl', 'wb') as f:
            pickle.dump(target_encoder, f)
            
        return variables_encoded

    def frequency_encoding(self):
        """
        Realiza codificación de frecuencia en las columnas especificadas en el diccionario de codificación.

        Returns:
        - dataframe: DataFrame de pandas, el DataFrame con codificación de frecuencia aplicada.
        """

        # accedemos a la clave de 'frequency' para poder extraer las columnas a las que que queramos aplicar Frequency Encoding. En caso de que no exista la clave, esta variable será una lista vacía
        col_encode = self.diccionario_encoding.get("frequency", [])

        # si hay contenido en la lista 
        if col_encode:

            # iteramos por cada una de las columnas a las que les queremos aplicar este tipo de encoding
            for categoria in col_encode:

                # calculamos las frecuencias de cada una de las categorías
                frecuencia = self.dataframe[categoria].value_counts(normalize=True)

                # mapeamos los valores obtenidos en el paso anterior, sobreescribiendo la columna original
                self.dataframe[categoria] = self.dataframe[categoria].map(frecuencia)
        
        return self.dataframe