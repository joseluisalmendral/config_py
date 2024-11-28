# Para gestionar el feature scaling
# -----------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import pickle

class FeatureScaling:
    def __init__(self, dataframe, lista_numericas):
        """
        Inicializa la clase con un dataframe y una lista de columnas numéricas.

        Parameters:
        dataframe (pd.DataFrame): El dataframe a escalar.
        lista_numericas (list): Lista de nombres de las columnas numéricas a escalar.
        """
        self.dataframe = dataframe
        self.lista_numericas = lista_numericas
        self.validate_columns()

    def validate_columns(self):
        """
        Verifica que todas las columnas en la lista de columnas numéricas existan en el dataframe.
        """
        missing_cols = [col for col in self.lista_numericas if col not in self.dataframe.columns]
        if missing_cols:
            raise ValueError(f"Las siguientes columnas no están en el dataframe: {missing_cols}")

    def scale_data(self, scaler):
        """
        Escala los datos utilizando el scaler proporcionado.

        Parameters:
        scaler: Un objeto de escalado de sklearn (MinMaxScaler, StandardScaler, RobustScaler).

        Returns:
        pd.DataFrame: Dataframe con los datos escalados.
        """
        datos_escalados = scaler.fit_transform(self.dataframe[self.lista_numericas])
        # Guardar el modelo
        with open('transformer_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        return pd.DataFrame(datos_escalados, columns=self.lista_numericas, index=self.dataframe.index)

    def min_max_scaler(self):
        """
        Aplica el escalado Min-Max a las columnas numéricas.

        Returns:
        pd.DataFrame: Dataframe con los datos escalados.
        """
        return self.scale_data(MinMaxScaler())


    def standard_scaler(self):
        """
        Aplica la estandarización a las columnas numéricas.

        Returns:
        pd.DataFrame: Dataframe con los datos estandarizados.
        """
        
        return self.scale_data(StandardScaler())

    def robust_scaler(self):
        """
        Aplica el escalado robusto a las columnas numéricas.

        Returns:
        pd.DataFrame: Dataframe con los datos escalados robustamente.
        """
        return self.scale_data(RobustScaler())

    def unir_datos_escalados(self, df_datos_escalados):
        """
        Actualiza el dataframe original con los datos escalados.

        Parameters:
        df_datos_escalados (pd.DataFrame): Dataframe con los datos escalados.

        Returns:
        pd.DataFrame: Dataframe original actualizado con los datos escalados.
        """
        self.dataframe[self.lista_numericas] = df_datos_escalados
        return self.dataframe
