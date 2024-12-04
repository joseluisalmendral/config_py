# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np
import pickle

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import plotly_express as px
from IPython.display import display, HTML

# Vigilar progreso bucles
# -----------------------------------------------------------------------
from tqdm import tqdm

# Machine Learning
# -----------------------------------------------------------------------
from sklearn.model_selection import train_test_split,GridSearchCV


# Para el modelado de los datos
# -----------------------------------------------------------------------
from sklearn.metrics import silhouette_score, pairwise_distances,  davies_bouldin_score


# Sacar número de clusters y métricas
# -----------------------------------------------------------------------
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score

# Modelos de clustering
# -----------------------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering


# Gestionar los warnings
# -----------------------------------------------------------------------
import warnings

# modificar el path
# -----------------------------------------------------------------------
import sys
import os

# Añade la ruta raíz del proyecto al sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)


# importar funciones de soporte
# -----------------------------------------------------------------------
from src.eda import soporte_preprocesamiento as sup_prep
from src.eda import soporte_nulos as sup_nul
from src.eda import soporte_outliers as sup_out
from src.clasificacion import soportefeaturescaling as sup_fea
from src.eda import soporte_encoding2 as sup_encod
from src.clasificacion import soporte_modelos_clasificacion as sup_models
from src.clustering import soporte_clustering as sup_clus

##aplicar configuraciones
#------------------------------------------------------------------------
warnings.filterwarnings('ignore')
pd.set_option('display.max_info_columns', 50)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format) #eliminamos la notacion cientifica

tqdm.pandas()