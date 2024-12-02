# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

import time
import psutil

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
import shap

# Para realizar la clasificación y la evaluación del modelo
# -----------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve
)

import xgboost as xgb
import pickle

# Para realizar cross validation
# -----------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import KBinsDiscretizer


class AnalisisModelosClasificacion:
    def __init__(self, dataframe, variable_dependiente, random_state=42):
        self.dataframe = dataframe
        self.variable_dependiente = variable_dependiente
        self.random_state = random_state
        self.X = dataframe.drop(variable_dependiente, axis=1)
        self.y = dataframe[variable_dependiente]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, train_size=0.8, random_state=self.random_state, shuffle=True
        )

        # Diccionario de modelos y resultados
        self.modelos = {
            "logistic_regression": LogisticRegression(random_state=self.random_state),
            "tree": DecisionTreeClassifier(random_state=self.random_state),
            "random_forest": RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            "gradient_boosting": GradientBoostingClassifier(random_state=self.random_state),
            "xgboost": xgb.XGBClassifier(random_state=self.random_state)
        }
        self.resultados = {nombre: {"mejor_modelo": None, "pred_train": None, "pred_test": None} for nombre in self.modelos}

    def ajustar_modelo(self, modelo_nombre, param_grid=None, scoring="accuracy", random_state=42, devolver_objeto=False, entrenamiento_final=False):
        """
        Ajusta el modelo seleccionado con GridSearchCV.
        Si entrenamiento_final=True, el modelo se entrena con todo el conjunto de datos (X, y).
        """
        if modelo_nombre not in self.modelos:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.modelos[modelo_nombre]

        # Parámetros predeterminados por modelo
        parametros_default = {
            "tree": {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "random_forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [2, 6, 8, 20, 12, 16],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "gradient_boosting": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 1.0]
            },
            "xgboost": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }

        if param_grid is None:
            param_grid = parametros_default.get(modelo_nombre, {})

        # Decidir los datos a usar según el parámetro `entrenamiento_final`
        if entrenamiento_final:
            print("\n **** Se está entrenando al modelo con TODO el conjunto de datos **** \n")
            X_datos = self.X
            y_datos = self.y
        else:
            X_datos = self.X_train
            y_datos = self.y_train

        if modelo_nombre == "logistic_regression":
            modelo_logistica = LogisticRegression(random_state=self.random_state)
            modelo_logistica.fit(X_datos, y_datos)

            if not entrenamiento_final:
                self.resultados[modelo_nombre]["pred_train"] = modelo_logistica.predict(self.X_train)
                self.resultados[modelo_nombre]["pred_test"] = modelo_logistica.predict(self.X_test)
            else:
                self.resultados[modelo_nombre]["pred_train"] = modelo_logistica.predict(self.X)
            self.resultados[modelo_nombre]["mejor_modelo"] = modelo_logistica

            if devolver_objeto:
                return modelo_logistica
            
        else:
            # Ajuste del modelo con GridSearchCV
            grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=8, scoring=scoring, n_jobs=-1, verbose=1)
            grid_search.fit(X_datos, y_datos)
            print(f"El mejor modelo es {grid_search.best_estimator_}")
            self.resultados[modelo_nombre]["mejor_modelo"] = grid_search.best_estimator_

            if not entrenamiento_final:
                self.resultados[modelo_nombre]["pred_train"] = grid_search.best_estimator_.predict(self.X_train)
                self.resultados[modelo_nombre]["pred_test"] = grid_search.best_estimator_.predict(self.X_test)
            else:
                self.resultados[modelo_nombre]["pred_train"] = grid_search.best_estimator_.predict(self.X)

            if devolver_objeto:
                return grid_search.best_estimator_


    def calcular_metricas(self, modelo_nombre, entrenamiento_final=False):
        """
        Calcula métricas de rendimiento para el modelo seleccionado, considerando si el modelo
        fue entrenado con el conjunto completo (entrenamiento_final=True) o con train/test split.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        pred_train = self.resultados[modelo_nombre]["pred_train"]
        pred_test = self.resultados[modelo_nombre]["pred_test"]

        if pred_train is None or (not entrenamiento_final and pred_test is None):
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular métricas.")
        
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]

        # Registrar tiempo de ejecución
        start_time = time.time()
        if hasattr(modelo, "predict_proba"):
            if entrenamiento_final:
                print("\n **** Se están mostrando las métricas para el entrenamiento del modelo con TODO el conjunto de datos **** \n")
                prob_train = modelo.predict_proba(self.X)[:, 1]
                prob_test = None
            else:
                prob_train = modelo.predict_proba(self.X_train)[:, 1]
                prob_test = modelo.predict_proba(self.X_test)[:, 1]
        else:
            prob_train = None
            prob_test = None
        elapsed_time = time.time() - start_time

        # Registrar núcleos utilizados
        num_nucleos = getattr(modelo, "n_jobs", psutil.cpu_count(logical=True))

        # Métricas para conjunto completo (entrenamiento_final=True)
        if entrenamiento_final:
            metricas_completas = {
                "accuracy": accuracy_score(self.y, pred_train),
                "precision": precision_score(self.y, pred_train, average='weighted', zero_division=0),
                "recall": recall_score(self.y, pred_train, average='weighted', zero_division=0),
                "f1": f1_score(self.y, pred_train, average='weighted', zero_division=0),
                "kappa": cohen_kappa_score(self.y, pred_train),
                "auc": roc_auc_score(self.y, prob_train) if prob_train is not None else None,
                "time_seconds": elapsed_time,
                "n_jobs": num_nucleos
            }
            return pd.DataFrame({"Conjunto completo": metricas_completas}).T

        # Métricas para conjunto de entrenamiento
        metricas_train = {
            'prediction': 'train',
            "accuracy": accuracy_score(self.y_train, pred_train),
            "precision": precision_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "recall": recall_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "f1": f1_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "kappa": cohen_kappa_score(self.y_train, pred_train),
            "auc": roc_auc_score(self.y_train, prob_train) if prob_train is not None else None,
            "time_seconds": elapsed_time,
            "n_jobs": num_nucleos
        }

        # Métricas para conjunto de prueba
        metricas_test = {
            'prediction': 'test',
            "accuracy": accuracy_score(self.y_test, pred_test),
            "precision": precision_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "recall": recall_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "f1": f1_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "kappa": cohen_kappa_score(self.y_test, pred_test),
            "auc": roc_auc_score(self.y_test, prob_test) if prob_test is not None else None,
            "time_seconds": elapsed_time,
            "n_jobs": num_nucleos
        }

        # Combinar métricas en un DataFrame
        return pd.DataFrame({"train": metricas_train, "test": metricas_test}).T


    def plot_matriz_confusion(self, modelo_nombre, figsize=(8, 6)):
        """
        Plotea la matriz de confusión para el modelo seleccionado.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")

        pred_test = self.resultados[modelo_nombre]["pred_test"]

        if pred_test is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular la matriz de confusión.")

        # Matriz de confusión
        matriz_conf = confusion_matrix(self.y_test, pred_test)
        plt.figure(figsize=figsize)
        sns.heatmap(matriz_conf, annot=True, fmt='g', cmap='Blues')
        plt.title(f"Matriz de Confusión ({modelo_nombre})")
        plt.xlabel("Predicción")
        plt.ylabel("Valor Real")
        plt.show()


    def importancia_predictores(self, modelo_nombre, figsize=(8, 4)):
        """
        Calcula y grafica la importancia de las características para el modelo seleccionado.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]
        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular importancia de características.")
        
        # Verificar si el modelo tiene importancia de características
        if hasattr(modelo, "feature_importances_"):
            importancia = modelo.feature_importances_
        elif modelo_nombre == "logistic_regression" and hasattr(modelo, "coef_"):
            importancia = modelo.coef_[0]
        else:
            print(f"El modelo '{modelo_nombre}' no soporta la importancia de características.")
            return
        
        # Crear DataFrame y graficar
        importancia_df = pd.DataFrame({
            "Feature": self.X.columns,
            "Importance": importancia
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=figsize)
        sns.barplot(x="Importance", y="Feature", data=importancia_df, palette="viridis")
        plt.title(f"Importancia de Características ({modelo_nombre})")
        plt.xlabel("Importancia")
        plt.ylabel("Características")
        plt.show()

    def plot_shap_summary(self, modelo_nombre, plot_size=(10, 5)):
        """
        Genera un SHAP summary plot para el modelo seleccionado.
        Maneja correctamente modelos de clasificación con múltiples clases.

        Parámetros:
            modelo_nombre (str): Nombre del modelo.
            plot_size (tuple): Tamaño de la figura (ancho, alto) en pulgadas.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")

        modelo = self.resultados[modelo_nombre]["mejor_modelo"]

        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de generar el SHAP plot.")

        # Usar TreeExplainer para modelos basados en árboles
        if modelo_nombre in ["tree", "random_forest", "gradient_boosting", "xgboost"]:
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(self.X_test)

            # Verificar si los SHAP values tienen múltiples clases (dimensión 3)
            if isinstance(shap_values, list):
                # Para modelos binarios, seleccionar SHAP values de la clase positiva
                shap_values = shap_values[1]
            elif len(shap_values.shape) == 3:
                # Para Decision Trees, seleccionar SHAP values de la clase positiva
                shap_values = shap_values[:, :, 1]
        else:
            # Usar el explicador genérico para otros modelos
            explainer = shap.Explainer(modelo, self.X_test, check_additivity=False)
            shap_values = explainer(self.X_test).values

        # Generar el summary plot con el tamaño personalizado
        shap.summary_plot(shap_values, self.X_test, feature_names=self.X.columns, plot_size=plot_size)



    def curva_roc(self, modelo_nombre):
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]
        if modelo is None:
            raise ValueError(f"Se debe ajustar el modelo '{modelo_nombre}' antes de calcular la curva ROC.")
        
        if not hasattr(modelo, "predict_proba"):
            raise ValueError(f"El modelo '{modelo_nombre}' no soporta la predicción de probabilidades.")
        
        # Get predicted probabilities for the positive class
        y_pred_test_prob = modelo.predict_proba(self.X_test)[:, 1]
        
        # Calculate ROC curve metrics
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_test_prob)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        sns.lineplot(x=fpr, y=tpr, color="blue", label="Modelo")
        sns.lineplot(x=[0, 1], y=[0, 1], color="grey", linestyle="--", label="Aleatorio")
        
        plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
        plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
        plt.title(f"Curva ROC: {modelo_nombre}")
        plt.legend(loc="lower right")
        plt.show()

    def filtrar_errores(self, modelo_nombre, tipo_error):
        """
        Filtra los errores de predicción del modelo especificado.

        Dependiendo del tipo de error indicado, devuelve las muestras clasificadas como
        falsos positivos o falsos negativos.

        Un falso positivo ocurre cuando el modelo predice la clase positiva (1),
        pero el valor real es negativo (0).
        Un falso negativo ocurre cuando el modelo predice la clase negativa (0),
        pero el valor real es positivo (1).

        Parámetros:
            modelo_nombre (str): Nombre del modelo cuyo rendimiento se evaluará.
            tipo_error (str): Tipo de error a filtrar. Debe ser "falsos_positivos" o "falsos_negativos".

        Devuelve:
            DataFrame: Contiene las muestras clasificadas como el tipo de error especificado,
            incluyendo las características originales y los valores reales/predichos.

        Excepciones:
            ValueError: Si el modelo no está entrenado, no es reconocido o si el tipo de error no es válido.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        # Obtener el modelo entrenado
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]
        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de filtrar errores.")
        
        if tipo_error not in ["fp", "fn"]:
            raise ValueError("El tipo de error debe ser 'fn' o 'fp'.")
        
        # Realizar predicciones
        y_pred = modelo.predict(self.X_test)
        
        # Crear un DataFrame para comparación
        resultados = pd.DataFrame({
            "real": self.y_test,
            "predicho": y_pred
        }, index=self.X_test.index)

        # Agregar los datos originales del conjunto de prueba para contexto
        resultados = pd.concat([self.dataframe.loc[resultados.index], resultados], axis=1)

        # Filtrar según el tipo de error
        if tipo_error == "fp":
            errores = resultados[(resultados["real"] == 0) & (resultados["predicho"] == 1)]
        elif tipo_error == "fn":
            errores = resultados[(resultados["real"] == 1) & (resultados["predicho"] == 0)]
        
        return errores
    
    def matrices_confusion(self, modelos, figsize=(12, 12)):

        num_models = len(modelos)
        cols = 2  
        rows = (num_models + 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() 

        for idx, modelo_nombre in enumerate(modelos):
            if modelo_nombre not in self.resultados:
                axes[idx].text(0.5, 0.5, f"Modelo '{modelo_nombre}' no encontrado", 
                            ha='center', va='center', fontsize=10)
                continue

            pred_test = self.resultados[modelo_nombre]["pred_test"]
            if pred_test is None:
                axes[idx].text(0.5, 0.5, f"Modelo '{modelo_nombre}' no ajustado", 
                            ha='center', va='center', fontsize=10)
                continue

            matriz_conf = confusion_matrix(self.y_test, pred_test)
            sns.heatmap(matriz_conf, annot=True, fmt='g', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f"Matriz ({modelo_nombre})")
            axes[idx].set_xlabel("Predicción")
            axes[idx].set_ylabel("Valor Real")

        # Quitar subplots
        for idx in range(num_models, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

    def curvas_roc(self, modelos, figsize=(12, 12)):

        num_models = len(modelos)
        cols = 2 
        rows = (num_models + 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()

        for idx, modelo_nombre in enumerate(modelos):
            if modelo_nombre not in self.resultados:
                axes[idx].text(0.5, 0.5, f"Modelo '{modelo_nombre}' no encontrado", 
                            ha='center', va='center', fontsize=10)
                continue

            modelo = self.resultados[modelo_nombre]["mejor_modelo"]
            if modelo is None or not hasattr(modelo, "predict_proba"):
                axes[idx].text(0.5, 0.5, f"Modelo '{modelo_nombre}' no ajustado o no soporta ROC", 
                            ha='center', va='center', fontsize=10)
                continue

            y_pred_test_prob = modelo.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_test_prob)

            axes[idx].plot(fpr, tpr, label="Modelo")
            axes[idx].plot([0, 1], [0, 1], linestyle="--", color="grey", label="Aleatorio")
            axes[idx].set_title(f"Curva ROC ({modelo_nombre})")
            axes[idx].set_xlabel("")
            axes[idx].set_ylabel("")
            axes[idx].legend()

        for idx in range(num_models, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()


    def curvas_roc_combinadas(self, modelos):

        if not hasattr(self, "X_train") or not hasattr(self, "X_test"):
            raise AttributeError("Train and test datasets are required (X_train, y_train, X_test, y_test).")

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        axes = axes.flatten()

        for dataset, ax in zip(["Train", "Test"], axes):
            for modelo_nombre in modelos:
                if modelo_nombre not in self.resultados:
                    print(f"Modelo '{modelo_nombre}' no encontrado.")
                    continue

                modelo = self.resultados[modelo_nombre]["mejor_modelo"]
                if modelo is None or not hasattr(modelo, "predict_proba"):
                    print(f"Modelo '{modelo_nombre}' no ajustado o no soporta predict_proba.")
                    continue

                # Select dataset based on train/test loop
                X = self.X_train if dataset == "Train" else self.X_test
                y = self.y_train if dataset == "Train" else self.y_test

                # Predict probabilities and calculate ROC
                y_pred_prob = modelo.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y, y_pred_prob)
                auc = roc_auc_score(y, y_pred_prob)

                # Plot ROC curve
                ax.plot(fpr, tpr, label=f"{modelo_nombre} (AUC = {auc:.2f})")

            ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Aleatorio")
            ax.set_title(f"Curva ROC - {dataset}")
            ax.set_xlabel("Tasa de Falsos Positivos")
            ax.set_ylabel("Tasa de Verdaderos Positivos")
            ax.legend()
            ax.grid()

        plt.tight_layout()
        plt.show()


    def shap_plots(self, modelos, plot_size=(10, 5), figsize=(15, 15)):

        num_modelos = len(modelos)
        cols = 2  # Number of columns for subplots
        rows = (num_modelos + 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()  # Flatten axes array for easy iteration

        for idx, modelo_nombre in enumerate(modelos):
            ax = axes[idx]

            if modelo_nombre not in self.resultados:
                ax.text(0.5, 0.5, f"Modelo '{modelo_nombre}' no encontrado", ha="center", va="center", fontsize=10)
                ax.axis("off")
                continue

            modelo = self.resultados[modelo_nombre]["mejor_modelo"]

            if modelo is None:
                ax.text(0.5, 0.5, f"Modelo '{modelo_nombre}' no ajustado", ha="center", va="center", fontsize=10)
                ax.axis("off")
                continue

            # Usar TreeExplainer para modelos basados en árboles
            if modelo_nombre in ["tree", "random_forest", "gradient_boosting", "xgboost"]:
                explainer = shap.TreeExplainer(modelo)
                shap_values = explainer.shap_values(self.X_test)

                # Verificar si los SHAP values tienen múltiples clases (dimensión 3)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification, use positive class
                elif len(shap_values.shape) == 3:
                    shap_values = shap_values[:, :, 1]
            else:
                # Usar el explicador genérico para otros modelos
                explainer = shap.Explainer(modelo, self.X_test, check_additivity=False)
                shap_values = explainer(self.X_test).values

            # Generar el summary plot en el subplot actual
            shap.summary_plot(shap_values, self.X_test, feature_names=self.X.columns, plot_size=plot_size, show=False)
            ax.set_title(f"SHAP Summary ({modelo_nombre})", fontsize=12)
            plt.sca(ax)  # Eje

        # Quitar subplots sin uso
        for idx in range(len(modelos), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

def revertir_datos_transformados(df_codificado, scaler_path, scaler_columns, original_columns, encoders_info):
    """
    Revierte las transformaciones de codificación y estandarización realizadas en un DataFrame.

    Parámetros:
        df_codificado (pd.DataFrame): DataFrame con las características codificadas y estandarizadas.
        scaler_path (str): Ruta al archivo pickle del scaler guardado (e.g., StandardScaler o MinMaxScaler).
        scaler_columns (list): Lista de columnas que fueron escaladas y necesitan ser revertidas.
        original_columns (list): Lista de columnas originales correspondientes a las escaladas.
        encoders_info (dict): Diccionario que mapea encoders a una lista de columnas y sus rutas pickle.
                             Formato: {"ruta_encoder.pkl": ["columna1", "columna2", ...], ...}

    Devuelve:
        pd.DataFrame: DataFrame con los datos revertidos a su estado original.
    """
    import pickle
    import numpy as np

    # Cargar el scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Revertir la estandarización solo para las columnas seleccionadas
    df_revertido = df_codificado.copy()
    if scaler_columns:
        try:
            df_revertido[original_columns] = scaler.inverse_transform(df_codificado[scaler_columns])
        except ValueError as e:
            raise ValueError(f"Error al revertir escalado: {e}. Verifique las columnas del DataFrame.") from e

    # Revertir codificación para las columnas categóricas
    for encoder_path, columnas in encoders_info.items():
        # Cargar el encoder correspondiente
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)

        # Si el encoder es TargetEncoder, usar mapeo interno
        if hasattr(encoder, "mapping_"):
            for columna in columnas:
                if columna in df_revertido.columns:
                    try:
                        # Ensure the mapping exists for the specific column
                        if columna not in encoder.mapping_:
                            raise ValueError(f"Mapping no encontrado para la columna '{columna}'.")
                        
                        # Retrieve and invert the mapping
                        mapping = encoder.mapping_[columna]
                        if mapping is None:
                            raise ValueError(f"Mapping vacío para la columna '{columna}'.")
                        
                        # Perform inverse mapping
                        inverse_mapping = {v: k for k, v in mapping.items()}
                        df_revertido[columna] = df_revertido[columna].map(inverse_mapping)
                    except KeyError as e:
                        raise ValueError(f"Error al revertir target encoding para la columna '{columna}': {e}.") from e

        # Si el encoder es OneHotEncoder, procesar múltiples columnas
        elif hasattr(encoder, "inverse_transform") and hasattr(encoder, "categories_"):
            try:
                # Extract only the columns that exist in the DataFrame
                existing_columns = [col for col in columnas if col in df_revertido.columns]
                
                # Ensure we have enough columns for inverse transformation
                if len(existing_columns) == 0:
                    raise ValueError(f"No matching columns found in DataFrame for {columnas}.")
                
                # Perform the inverse transformation
                ohe_data = df_revertido[existing_columns].values
                reverted_col = encoder.inverse_transform(ohe_data)
                
                # Add the reverted column back to the DataFrame
                reverted_col_name = existing_columns[0].split('_')[0] + "_reverted"
                df_revertido[reverted_col_name] = reverted_col

                # Drop the original one-hot encoded columns
                df_revertido.drop(columns=existing_columns, axis=1, inplace=True)

            except Exception as e:
                raise ValueError(f"Error al revertir codificación para las columnas {columnas}: {e}.") from e

        # Si el encoder es LabelEncoder u otro similar
        else:
            for columna in columnas:
                if columna in df_revertido.columns:
                    try:
                        df_revertido[columna] = encoder.inverse_transform(df_revertido[columna].astype(int))
                    except ValueError as e:
                        raise ValueError(f"Error al revertir codificación para la columna '{columna}': {e}.") from e
    
    return df_revertido

# Función para asignar colores
def color_filas_por_modelo(row):
    if row["modelo"] == "tree":
        return ["background-color: #e6b3e0; color: black"] * len(row)  
    
    elif row["modelo"] == "random_forest":
        return ["background-color: #c2f0c2; color: black"] * len(row) 

    elif row["modelo"] == "gradient_boost":
        return ["background-color: #ffd9b3; color: black"] * len(row)  

    elif row["modelo"] == "x_gradient_boost":
        return ["background-color: #f7b3c2; color: black"] * len(row)  

    elif row["modelo"] == "logistic_regression":
        return ["background-color: #b3d1ff; color: black"] * len(row)  
    
    return ["color: black"] * len(row)