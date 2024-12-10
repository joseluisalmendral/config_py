# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otras utilidades
# -----------------------------------------------------------------------
import math

# Para las visualizaciones
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Preprocesado y modelado
# -----------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer


# Sacar número de clusters y métricas
# -----------------------------------------------------------------------
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

# Modelos de clustering
# -----------------------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

# Para el modelado de los datos
# -----------------------------------------------------------------------
from sklearn.metrics import silhouette_score, pairwise_distances,  davies_bouldin_score


# Sacar número de clusters y métricas
# -----------------------------------------------------------------------
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score

from sklearn_extra.cluster import KMedoids

# Para visualizar los dendrogramas
# -----------------------------------------------------------------------
import scipy.cluster.hierarchy as sch

class Exploracion:
    """
    Clase para realizar la exploración y visualización de datos en un DataFrame.

    Atributos:
    dataframe : pd.DataFrame
        El conjunto de datos a ser explorado y visualizado.
    """

    def __init__(self, dataframe):
        """
        Inicializa la clase Exploracion con un DataFrame.

        Params:
            - dataframe : pd.DataFrame. El DataFrame que contiene los datos a ser explorados.
        """
        self.dataframe = dataframe
        
    
    def explorar_datos(self):
        """
        Realiza un análisis exploratorio de un DataFrame.

        Params:
            - Ninguno.

        Returns:
            - None.
        """
        print("5 registros aleatorios:")
        display(self.dataframe.sample(5))
        print("\n")

        print("Información general del DataFrame:")
        print(self.dataframe.info())
        print("\n")

        print("Duplicados en el DataFrame:")
        print(self.dataframe.duplicated().sum())
        print("\n")

        print("Estadísticas descriptivas de las columnas numéricas:")
        display(self.dataframe.describe().T)
        print("\n")

        print("Estadísticas descriptivas de las columnas categóricas:")
        categorical_columns = self.dataframe.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            display(self.dataframe[categorical_columns].describe().T)
        else:
            print("No hay columnas categóricas en el DataFrame.")
        print("\n")
        
        print("Número de valores nulos por columna:")
        print(self.dataframe.isnull().sum())
        print("\n")
        
        if len(categorical_columns) > 0:
            print("Distribución de valores categóricos:")
            for col in categorical_columns:
                print(f"\nColumna: {col}")
                print(self.dataframe[col].value_counts())
        
        print("Matriz de correlación entre variables numéricas:")
        display(self.dataframe.corr(numeric_only=True))
        print("\n")

    def visualizar_numericas(self):
        """
        Genera histogramas, boxplots y gráficos de dispersión para las variables numéricas del DataFrame.

        Params:
            - Ninguno.

        Returns:
            - None.
        """
        columns = self.dataframe.select_dtypes(include=np.number).columns

        # Histogramas
        fig, axes = plt.subplots(nrows=math.ceil(len(columns)/2), ncols=2, figsize=(21, 13))
        axes = axes.flat
        plt.suptitle("Distribución de las variables numéricas", fontsize=24)
        for indice, columna in enumerate(columns):
            sns.histplot(x=columna, data=self.dataframe, ax=axes[indice], kde=True, color="#F2C349")

        if len(columns) % 2 != 0:
            fig.delaxes(axes[-1])

        plt.tight_layout()

        # Boxplots
        fig, axes = plt.subplots(nrows=math.ceil(len(columns)/2), ncols=2, figsize=(19, 11))
        axes = axes.flat
        plt.suptitle("Boxplots de las variables numéricas", fontsize=24)
        for indice, columna in enumerate(columns):
            sns.boxplot(x=columna, data=self.dataframe, ax=axes[indice], color="#F2C349", flierprops={'markersize': 4, 'markerfacecolor': 'cyan'})
        if len(columns) % 2 != 0:
            fig.delaxes(axes[-1])
        plt.tight_layout()
    
    def visualizar_categoricas(self):
        """
        Genera gráficos de barras (count plots) para las variables categóricas del DataFrame.

        Params:
            - Ninguno.

        Returns:
            - None.
        """
        categorical_columns = self.dataframe.select_dtypes(include=['object', 'category']).columns

        if len(categorical_columns) > 0:
            try:
                _, axes = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(15, 5 * len(categorical_columns)))
                axes = axes.flat
                plt.suptitle("Distribución de las variables categóricas", fontsize=24)
                for indice, columna in enumerate(categorical_columns):
                    sns.countplot(data=self.dataframe, x=columna, ax=axes[indice])
                    axes[indice].set_title(f'Distribución de {columna}', fontsize=20)
                    axes[indice].set_xlabel(columna, fontsize=16)
                    axes[indice].set_ylabel('Conteo', fontsize=16)
                plt.tight_layout()
            except: 
                sns.countplot(data=self.dataframe, x=categorical_columns[0])
                plt.title(f'Distribución de {categorical_columns[0]}', fontsize=20)
                plt.xlabel(categorical_columns[0], fontsize=16)
                plt.ylabel('Conteo', fontsize=16)
        else:
            print("No hay columnas categóricas en el DataFrame.")

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
                    _, axes = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(10, 5 * len(categorical_columns)))
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
        else:
            print("No hay columnas categóricas en el DataFrame.")

    def correlacion(self, metodo="pearson", tamanio=(14, 8)):
        """
        Genera un heatmap de la matriz de correlación de las variables numéricas del DataFrame.

        Params:
            - metodo : str, optional, default: "pearson". Método para calcular la correlación.
            - tamanio : tuple of int, optional, default: (14, 8). Tamaño de la figura del heatmap.

        Returns:
            - None.
        """
        plt.figure(figsize=tamanio)
        mask = np.triu(np.ones_like(self.dataframe.corr(numeric_only=True), dtype=np.bool_))
        sns.heatmap(self.dataframe.corr(numeric_only=True, method=metodo), annot=True, cmap='viridis', vmax=1, vmin=-1, mask=mask)
        plt.title("Correlación de las variables numéricas", fontsize=24)



class Preprocesado:
    """
    Clase para realizar preprocesamiento de datos en un DataFrame.

    Atributos:
        - dataframe : pd.DataFrame. El conjunto de datos a ser preprocesado.
    """
    
    def __init__(self, dataframe):
        """
        Inicializa la clase Preprocesado con un DataFrame.

        Params:
            - dataframe : pd.DataFrame. El DataFrame que contiene los datos a ser preprocesados.
        """
        self.dataframe = dataframe
        self.escaladores = {
                            'minmax': MinMaxScaler(),
                            'normalizer': Normalizer(),
                            'standard': StandardScaler(),
                            'robust': RobustScaler()
                        }

    def estandarizar(self, escalador = 'standard'):
        """
        Estandariza las columnas numéricas del DataFrame.

        Este método ajusta y transforma las columnas numéricas del DataFrame utilizando `StandardScaler` para que
        tengan media 0 y desviación estándar 1.

        Returns:
            - pd.DataFrame. El DataFrame con las columnas numéricas estandarizadas.
        """
        # Sacamos el nombre de las columnas numéricas
        col_numericas = self.dataframe.select_dtypes(include=[np.number, 'number']).columns

        # Inicializamos el escalador para estandarizar los datos
        scaler = self.escaladores[escalador]

        # Ajustamos los datos y los transformamos
        X_scaled = scaler.fit_transform(self.dataframe[col_numericas])

        # Sobreescribimos los valores de las columnas en el DataFrame
        self.dataframe[col_numericas] = X_scaled

        return self.dataframe
    
    def codificar(self):
        """
        Codifica las columnas categóricas del DataFrame.

        Este método reemplaza los valores de las columnas categóricas por sus frecuencias relativas dentro de cada
        columna.

        Returns:
            - pd.DataFrame. El DataFrame con las columnas categóricas codificadas.
        """
        # Sacamos el nombre de las columnas categóricas
        col_categoricas = self.dataframe.select_dtypes(include=["category", "object"]).columns

        # Iteramos por cada una de las columnas categóricas para aplicar el encoding
        for categoria in col_categoricas:
            # Calculamos las frecuencias de cada una de las categorías
            frecuencia = self.dataframe[categoria].value_counts(normalize=True)

            # Mapeamos los valores obtenidos en el paso anterior, sobreescribiendo la columna original
            self.dataframe[categoria] = self.dataframe[categoria].map(frecuencia)

        return self.dataframe


class Clustering:
    """
    Clase para realizar varios métodos de clustering en un DataFrame.

    Atributos:
        - dataframe : pd.DataFrame. El conjunto de datos sobre el cual se aplicarán los métodos de clustering.
    """
    
    def __init__(self, dataframe):
        """
        Inicializa la clase Clustering con un DataFrame.

        Params:
            - dataframe : pd.DataFrame. El DataFrame que contiene los datos a los que se les aplicarán los métodos de clustering.
        """
        self.dataframe = dataframe
    
    def sacar_clusters_kmeans(self, n_clusters=(2, 15)):
        """
        Utiliza KMeans y KElbowVisualizer para determinar el número óptimo de clusters basado en la métrica de silhouette.

        Params:
            - n_clusters : tuple of int, optional, default: (2, 15). Rango de número de clusters a probar.
        
        Returns:
            None
        """
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=n_clusters, metric='silhouette')
        visualizer.fit(self.dataframe)
        visualizer.show()
    
    def modelo_kmeans(self, dataframe_original, num_clusters):
        """
        Aplica KMeans al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - num_clusters : int. Número de clusters a formar.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        km_fit = kmeans.fit(self.dataframe)
        labels = km_fit.labels_
        dataframe_original["clusters_kmeans"] = labels.astype(str)
        return dataframe_original, labels
    
    def visualizar_dendrogramas(self, lista_metodos=["average", "complete", "ward", "single"]):
        """
        Genera y visualiza dendrogramas para el conjunto de datos utilizando diferentes métodos de distancias.

        Params:
            - lista_metodos : list of str, optional, default: ["average", "complete", "ward"]. Lista de métodos para calcular las distancias entre los clusters. Cada método generará un dendrograma
                en un subplot diferente.

        Returns:
            None
        """
        _, axes = plt.subplots(nrows=1, ncols=len(lista_metodos), figsize=(20, 8))
        axes = axes.flat

        for indice, metodo in enumerate(lista_metodos):
            sch.dendrogram(sch.linkage(self.dataframe, method=metodo),
                           labels=self.dataframe.index, 
                           leaf_rotation=90, leaf_font_size=4,
                           ax=axes[indice])
            axes[indice].set_title(f'Dendrograma usando {metodo}')
            axes[indice].set_xlabel('Muestras')
            axes[indice].set_ylabel('Distancias')

    def visualizar_radar_plot(self, clustered_df: pd.DataFrame, columna_cluster: str, variables: list):
        
        # Agrupar por cluster y calcular la media
        cluster_means = clustered_df.groupby(columna_cluster)[variables].mean()

        # Repetir la primera columna al final para cerrar el radar
        cluster_means = pd.concat([cluster_means, cluster_means.iloc[:, 0:1]], axis=1)

        # Crear los ángulos para el radar plot
        num_vars = len(variables)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el gráfico

        # Crear el radar plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Dibujar un gráfico para cada cluster
        for i, row in cluster_means.iterrows():
            ax.plot(angles, row, label=f'Cluster {i}')
            ax.fill(angles, row, alpha=0.25)

        # Configurar etiquetas de los ejes
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(variables)

        # Añadir leyenda y título
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Radar Plot de los Clusters', size=16)
        plt.show()


    def calcular_configuraciones_vinculacion_distancia(self, df_ready_to_cluster: pd.DataFrame):
        # Lista de métricas de distancia a evaluar
        metricas = ['euclidean', 'manhattan', 'cosine']
        n_clusters_range = range(3, 6)  # Probar con 2, 3, 4 y 5 clusters

        # Diccionario para guardar resultados
        resultados = {'métrica': [], 'modelo': [], 'silhouette_score': [], 'clusters': []}

        # Probar KMeans y KMedoids con diferentes métricas y números de clusters
        for metrica in metricas:
            for n_clusters in n_clusters_range:
                # Preprocesar distancias si no es Euclidiana
                if metrica != 'euclidean':
                    distancias = pairwise_distances(df_ready_to_cluster, metric=metrica)
                else:
                    distancias = df_ready_to_cluster

                # KMeans
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                kmeans.fit(distancias)
                labels_kmeans = kmeans.labels_

                # Calcular índice de silueta para KMeans
                try:
                    sil_kmeans = silhouette_score(distancias, labels_kmeans, metric=metrica)
                except ValueError:  # Manejar errores en métricas incompatibles
                    sil_kmeans = None
                resultados['métrica'].append(metrica)
                resultados['modelo'].append('KMeans')
                resultados['silhouette_score'].append(sil_kmeans)
                resultados['clusters'].append(n_clusters)

                # KMedoids
                kmedoids = KMedoids(n_clusters=n_clusters, metric=metrica, random_state=42)
                kmedoids.fit(df_ready_to_cluster)
                labels_kmedoids = kmedoids.labels_

                # Calcular índice de silueta para KMedoids
                try:
                    sil_kmedoids = silhouette_score(df_ready_to_cluster, labels_kmedoids, metric=metrica)
                except ValueError:  # Manejar errores en métricas incompatibles
                    sil_kmedoids = None
                resultados['métrica'].append(metrica)
                resultados['modelo'].append('KMedoids')
                resultados['silhouette_score'].append(sil_kmedoids)
                resultados['clusters'].append(n_clusters)

        # Convertir resultados a DataFrame
        resultados_df = pd.DataFrame(resultados)

        # Mostrar los mejores resultados
        resultados_df.sort_values(by='silhouette_score', ascending=False, inplace=True)
        display(resultados_df)
        

    
    def modelo_aglomerativo(self, num_clusters, metodo_distancias, dataframe_original):
        """
        Aplica clustering aglomerativo al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - num_clusters : int. Número de clusters a formar.
            - metodo_distancias : str. Método para calcular las distancias entre los clusters.
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        modelo = AgglomerativeClustering(
            linkage=metodo_distancias,
            distance_threshold=None,
            n_clusters=num_clusters
        )
        aglo_fit = modelo.fit(self.dataframe)
        labels = aglo_fit.labels_
        dataframe_original["clusters_agglomerative"] = labels.astype(str)
        return dataframe_original
    

    def calcular_confs_vinculacion_distancia_aglomerativo(self, df_to_cluster: pd.DataFrame):
        # Configuraciones de vinculación y métricas de distancia
        linkage_methods = [ 'complete',  'ward']
        distance_metrics = ['euclidean', 'cosine', 'chebyshev']

        # Crear un DataFrame para almacenar los resultados
        results = []

        # Suponiendo que tienes un DataFrame llamado df_copia
        # Aquí df_copia debería ser tu conjunto de datos
        # Asegúrate de que esté preprocesado adecuadamente (normalizado si es necesario)

        for linkage_method in linkage_methods:
            for metric in distance_metrics:
                for cluster in range(3,10):
                    try:
                        # Configurar el modelo de AgglomerativeClustering
                        modelo = AgglomerativeClustering(
                            linkage=linkage_method,
                            metric=metric,  
                            distance_threshold=None,  # Para buscar n_clusters
                            n_clusters=cluster, # Cambia esto según tu análisis
                        )
                        
                        # Ajustar el modelo
                        labels = modelo.fit_predict(df_to_cluster)

                        # Calcular métricas si hay más de un cluster
                        if len(np.unique(labels)) > 1:
                            # Silhouette Score
                            silhouette_avg = silhouette_score(df_to_cluster, labels, metric=metric)

                            # Davies-Bouldin Index
                            db_score = davies_bouldin_score(df_to_cluster, labels)

                            
                            # Cardinalidad (tamaño de cada cluster)
                            cluster_cardinality = {cluster: sum(labels == cluster) for cluster in np.unique(labels)}
                        else:
                            inertia = float('inf')
                            cluster_cardinality = {'Cluster único': len(df_to_cluster)}

                        # Almacenar resultados
                        results.append({
                            'linkage': linkage_method,
                            'metric': metric,
                            'silhouette_score': silhouette_avg,
                            'davies_bouldin_index': db_score,
                            'cluster_cardinality': cluster_cardinality,
                            'n_cluster': cluster
                        })

                    except Exception as e:
                        print(f"Error con linkage={linkage_method}, metric={metric}: {e}")

        # Crear DataFrame de resultados
        results_df = pd.DataFrame(results)

        # Mostrar resultados ordenados por silhouette_score
        results_df = results_df.sort_values(by='silhouette_score', ascending=False)

        # Mostrar el DataFrame
        display(results_df.head(20))




    
    def modelo_divisivo(self, dataframe_original, threshold=0.5, max_clusters=5):
        """
        Implementa el clustering jerárquico divisivo.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - threshold : float, optional, default: 0.5. Umbral para decidir cuándo dividir un cluster.
            - max_clusters : int, optional, default: 5. Número máximo de clusters deseados.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de los clusters.
        """
        def divisive_clustering(data, current_cluster, cluster_labels):
            # Si el número de clusters actuales es mayor o igual al máximo permitido, detener la división
            if len(set(current_cluster)) >= max_clusters:
                return current_cluster

            # Aplicar KMeans con 2 clusters
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(data)
            labels = kmeans.labels_

            # Calcular la métrica de silueta para evaluar la calidad del clustering
            silhouette_avg = silhouette_score(data, labels)

            # Si la calidad del clustering es menor que el umbral o si el número de clusters excede el máximo, detener la división
            if silhouette_avg < threshold or len(set(current_cluster)) + 1 > max_clusters:
                return current_cluster

            # Crear nuevas etiquetas de clusters
            new_cluster_labels = current_cluster.copy()
            max_label = max(current_cluster)

            # Asignar nuevas etiquetas incrementadas para cada subcluster
            for label in set(labels):
                cluster_indices = np.where(labels == label)[0]
                new_label = max_label + 1 + label
                new_cluster_labels[cluster_indices] = new_label

            # Aplicar recursión para seguir dividiendo los subclusters
            for new_label in set(new_cluster_labels):
                cluster_indices = np.where(new_cluster_labels == new_label)[0]
                new_cluster_labels = divisive_clustering(data[cluster_indices], new_cluster_labels, new_cluster_labels)

            return new_cluster_labels

        # Inicializar las etiquetas de clusters con ceros
        initial_labels = np.zeros(len(self.dataframe))

        # Llamar a la función recursiva para iniciar el clustering divisivo
        final_labels = divisive_clustering(self.dataframe.values, initial_labels, initial_labels)

        # Añadir las etiquetas de clusters al DataFrame original
        dataframe_original["clusters_divisive"] = final_labels.astype(int).astype(str)

        return dataframe_original

    def modelo_espectral(self, dataframe_original, n_clusters=3, assign_labels='kmeans'):
        """
        Aplica clustering espectral al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - n_clusters : int, optional, default: 3. Número de clusters a formar.
            - assign_labels : str, optional, default: 'kmeans'. Método para asignar etiquetas a los puntos. Puede ser 'kmeans' o 'discretize'.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        spectral = SpectralClustering(n_clusters=n_clusters, assign_labels=assign_labels, random_state=0)
        labels = spectral.fit_predict(self.dataframe)
        dataframe_original["clusters_spectral"] = labels.astype(str)
        return dataframe_original
    
    def modelo_dbscan(self, dataframe_original, eps_values=[0.5, 1.0, 1.5], min_samples_values=[3, 2, 1]):
        """
        Aplica DBSCAN al DataFrame y genera clusters con diferentes combinaciones de parámetros. 
        Evalúa las métricas de calidad y retorna el DataFrame original con etiquetas de clusters.

        Parámetros:
        -----------
        dataframe_original : pd.DataFrame
            DataFrame original al que se le añadirán las etiquetas de clusters.
        eps_values : list of float, optional, default: [0.5, 1.0, 1.5]
            Lista de valores para el parámetro eps de DBSCAN.
        min_samples_values : list of int, optional, default: [3, 2, 1]
            Lista de valores para el parámetro min_samples de DBSCAN.

        Retorna:
        --------
        dataframe_original : pd.DataFrame
            DataFrame con una nueva columna `clusters_dbscan` que contiene las etiquetas de los clusters.
        metrics_df_dbscan : pd.DataFrame
            DataFrame con las métricas de evaluación para cada combinación de parámetros.
        """
        # Validación de entradas
        if not eps_values or not min_samples_values:
            raise ValueError("Las listas eps_values y min_samples_values no pueden estar vacías.")

        # Inicializar variables para almacenar el mejor modelo
        best_eps = None
        best_min_samples = None
        best_silhouette = float("-inf")
        metrics_results_dbscan = []

        # Iterar sobre combinaciones de parámetros
        for eps in tqdm(eps_values, desc=f"Iterando sobre eps y min_samples"):
            for min_samples in min_samples_values:
                try:
                    # Aplicar DBSCAN
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(dataframe_original)

                    # Evaluar solo si hay más de un cluster
                    if len(set(labels)) > 1 and len(set(labels)) < len(labels):
                        silhouette = silhouette_score(dataframe_original, labels)
                        davies_bouldin = davies_bouldin_score(dataframe_original, labels)

                        # Cardinalidad de los clusters
                        unique, counts = np.unique(labels, return_counts=True)
                        cardinalidad = dict(zip(unique, counts))

                        # Guardar resultados
                        metrics_results_dbscan.append({
                            "eps": eps,
                            "min_samples": min_samples,
                            "silhouette_score": silhouette,
                            "davies_bouldin_score": davies_bouldin,
                            "cardinality": cardinalidad
                        })

                        # Actualizar el mejor modelo
                        if silhouette > best_silhouette:
                            best_silhouette = silhouette
                            best_eps = eps
                            best_min_samples = min_samples
                except Exception as e:
                    print(f"Error con eps={eps}, min_samples={min_samples}: {e}")

        # Crear un DataFrame con las métricas
        metrics_df_dbscan = pd.DataFrame(metrics_results_dbscan).sort_values(by="silhouette_score", ascending=False)

        # Mostrar resultados
        display(metrics_df_dbscan)

        # Aplicar DBSCAN con los mejores parámetros
        if best_eps is not None and best_min_samples is not None:
            best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
            dataframe_original["clusters_dbscan"] = best_dbscan.fit_predict(dataframe_original)
            print(f"Mejor modelo: eps={best_eps}, min_samples={best_min_samples}, silhouette_score={best_silhouette:.4f}")
        else:
            print("No se encontraron clusters válidos con los parámetros proporcionados.")
            dataframe_original["clusters_dbscan"] = -1  # Asignar ruido si no se encontraron clusters

        cantidad_clusters = dataframe_original["clusters_dbscan"].nunique()
        print(f"Se han generado {cantidad_clusters} clusters (incluyendo outliers, si los hay).")

        return dataframe_original, metrics_df_dbscan
    
    

    def calcular_metricas(self, labels: np.ndarray):
        """
        Calcula métricas de evaluación del clustering.
        """
        if len(set(labels)) <= 1:
            raise ValueError("El clustering debe tener al menos 2 clusters para calcular las métricas.")

        silhouette = silhouette_score(self.dataframe, labels)
        davies_bouldin = davies_bouldin_score(self.dataframe, labels)

        unique, counts = np.unique(labels, return_counts=True)
        cardinalidad = dict(zip(unique, counts))

        return pd.DataFrame({
            "silhouette_score": silhouette,
            "davies_bouldin_index": davies_bouldin,
            "cardinalidad": cardinalidad
        }, index = [0])
    

    def plot_clusters(self, df_cluster, columna_cluster):
        columnas_plot = df_cluster.columns.drop(columna_cluster)

        ncols = math.ceil(len(columnas_plot) / 2)
        nrows = 2

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 8))
        axes = axes.flat

        for indice, columna in enumerate(columnas_plot):
            df_group = df_cluster.groupby(columna_cluster)[columna].mean().reset_index()
            sns.barplot(x=columna_cluster, y=columna, data=df_group, ax=axes[indice], palette="coolwarm")
            axes[indice].set_title(columna)  

        if len(columnas_plot) % 2 == 1: 
            fig.delaxes(axes[-1]) 

        plt.tight_layout()
        plt.show()

    
    def plot_completo_clusters(self, data, x, y=None, dateplot="month", hue=None):
        """
        Genera múltiples gráficos de las columnas seleccionadas de un DataFrame, con la opción 
        de usar una variable como "hue" para diferenciar grupos o categorías.

        Créditos:
        -----------
        Como base, he tomado una función del chino murciano más majo del mundo. Gracias <3 @https://github.com/yanruwu

        Parámetros:
        -----------
        data : pd.DataFrame
            El DataFrame que contiene los datos para graficar.
        x : str o list
            Columnas del DataFrame a graficar. Puede ser un string (columna única) o una lista 
            de nombres de columnas.
        y : str, opcional
            Columna del DataFrame para usar como variable dependiente en los gráficos (para gráficos
            como scatterplot, boxplot, etc.). Si no se especifica, se grafican las columnas de `x` 
            individualmente.
        dateplot : {"month", "year"}, opcional, default="month"
            Especifica cómo manejar las columnas de tipo fecha. Si es "month", las fechas se 
            agruparán por mes. Si es "year", se agruparán por año.
        hue : str, opcional
            Columna del DataFrame que se usará como "hue" para diferenciar colores en los gráficos. 
            Esto es útil para comparar grupos o clusters.

        Salida:
        -------
        Gráficos generados:
            - Para columnas numéricas: histogramas con la opción de kde (curva de densidad).
            - Para columnas categóricas: gráficos de conteo (countplot).
            - Para columnas de fecha: gráficos de conteo agrupados por mes o año.
            - Para combinaciones de columnas con `y`:
                * Numérico vs Numérico: scatterplot.
                * Categórico vs Numérico: boxplot.
                * Fecha vs Numérico: lineplot.

        Notas:
        ------
        - Si no se especifica `y`, la función genera gráficos individuales para cada columna en `x`.
        - Si se especifica `hue`, los gráficos mostrarán los datos diferenciados por la categoría 
        especificada en `hue`.

        Ejemplo:
        --------
        # Graficar todas las columnas con los clusters como hue
        plot_completo_clusters(data=df, x=df.columns, hue="clusters_dbscan")

        # Graficar columnas específicas
        plot_completo_clusters(data=df, x=["edad", "ingresos"], y="gasto", hue="segmento")

        """

        if type(x) == str:
            x = [x] #convertir x en lista

        # Separar tipos de datos
        num_cols = data[x].select_dtypes("number")
        cat_cols = data[x].select_dtypes("O", "category")
        date_cols = data[x].select_dtypes("datetime")

        if not y:
            nrows = 2
            ncols = math.ceil(len(x) / 2)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 15), dpi=130)
            axes = axes.flat
            for i, col in enumerate(x):
                if col in num_cols:
                    sns.histplot(data=data, x=col, hue=hue, ax=axes[i], bins=20, kde=True)
                    axes[i].set_title(col)
                elif col in cat_cols:
                    sns.countplot(data=data, x=col, hue=hue, ax=axes[i])
                    axes[i].tick_params(axis='x', rotation=90)
                    axes[i].set_title(col)
                elif col in date_cols:
                    sns.countplot(
                        data=data,
                        x=data[col].apply(lambda x: x.year) if dateplot == "year" else data[col].apply(lambda x: x.month),
                        hue=hue,
                        ax=axes[i]
                    )
                    axes[i].tick_params(axis='x', rotation=90)
                    axes[i].set_title(f"{col}_{dateplot}")
                else:
                    print(f"Advertencia: No se pudo graficar '{col}' ya que no es de un tipo soportado o la combinación no es válida.")
                
                axes[i].set_xlabel("")
                
            for j in range(len(x), len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()

        else:
            nrows = math.ceil(len(x) / 2)
            ncols = 2
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 25), dpi=130)
            axes = axes.flat

            for i, col in enumerate(x):
                # Caso 1: Numérico vs Numérico -> Scatterplot
                if col in num_cols and y in data.select_dtypes("number"):
                    sns.scatterplot(data=data, x=col, y=y, hue=hue, ax=axes[i])
                    axes[i].set_title(f'{col} vs {y}')
                # Caso 2: Categórico vs Numérico -> Boxplot o Violinplot
                elif col in cat_cols and y in data.select_dtypes("number"):
                    sns.boxplot(data=data, x=col, y=y, hue=hue, ax=axes[i])
                    axes[i].tick_params(axis='x', rotation=90)
                    axes[i].set_title(f'{col} vs {y}')
                # Caso 3: Fecha vs Numérico -> Lineplot
                elif col in date_cols and y in data.select_dtypes("number"):
                    date_data = data[col].dt.year if dateplot == "year" else data[col].dt.month
                    sns.lineplot(x=date_data, y=data[y], hue=hue, ax=axes[i])
                    axes[i].set_title(f'{col}_{dateplot} vs {y}')
                    axes[i].tick_params(axis='x', rotation=90)
                else:
                    print(f"Advertencia: No se pudo graficar '{col}' frente a '{y}' ya que no es de un tipo soportado o la combinación no es válida.")

                axes[i].set_xlabel("")
            
            for j in range(len(x), len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()

    
    def plot_epsilon(self, dataframe, k=10, figsize=(10, 6), ylim=(0, 0.2)):
        """
        Genera un gráfico de k-distancias para determinar el valor óptimo de epsilon (eps)
        al usar el algoritmo DBSCAN, y calcula numéricamente el valor de epsilon basado 
        en el máximo gradiente (codo).

        Parámetros:
        -----------
        dataframe : pd.DataFrame o np.ndarray
            Conjunto de datos a analizar. Cada fila es un punto, y las columnas son las características.
        k : int, opcional, default=10
            Número de vecinos a considerar. Generalmente se recomienda usar un valor igual al
            número de variables por 2, o ligeramente mayor.
        figsize : tuple, opcional, default=(10, 6)
            Tamaño de la figura del gráfico (ancho, alto).
        ylim : tuple, opcional, default=(0, 0.2)
            Límite en el eje y para enfocar la visualización de las distancias.

        Salida:
        -------
        Gráfico de k-distancias:
            - El eje x representa los puntos ordenados por distancia al k-ésimo vecino más cercano.
            - El eje y muestra la distancia al k-ésimo vecino más cercano.
            - La línea vertical roja indica el epsilon recomendado basado en el codo.

        Retorno:
        --------
        eps_value : float
            Valor recomendado para epsilon (eps).

        Notas:
        ------
        - Es importante escalar los datos antes de usar esta función, especialmente si las características 
        tienen diferentes rangos.

        Ejemplo:
        --------
        # Calcular y graficar distancias para encontrar eps
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(dataframe)
        self.plot_epsilon(dataframe=scaled_data, k=10, figsize=(12, 8), ylim=(0, 0.5))
        """
        # Paso 1: Encontrar a los vecinos más cercanos
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(dataframe)
        
        # Paso 2: calcular las distancias entre vecinos
        distances, indices = neighbors_fit.kneighbors(dataframe)
        k_distances = distances[:, k - 1]  # k-th nearest neighbor distances

        # Paso 3: Ordenar distancias
        k_distances = np.sort(k_distances)
        
        # Paso 4: Calcular el máximo gradiente (codo)
        gradients = np.diff(k_distances)  # Derivada numérica
        max_gradient_index = np.argmax(gradients)  # Índice con el mayor gradiente
        eps_value = k_distances[max_gradient_index]  # Valor de epsilon recomendado

        # Paso 5: Graficar el k-distance plot
        plt.figure(figsize=figsize)
        plt.plot(k_distances, label="distancias 'k'")
        plt.axvline(x=max_gradient_index, color='red', linestyle='--', label=f"Epsilon = {eps_value:.4f}")
        plt.ylim(ylim)
        plt.title(f"Plot K-Distance con k={k}")
        plt.xlabel("Puntos ordenados por distancia")
        plt.ylabel(f"Distancia al {k}-ésimo vecino")
        plt.legend()
        plt.grid()
        plt.show()

        # Retornar el valor de epsilon
        return eps_value