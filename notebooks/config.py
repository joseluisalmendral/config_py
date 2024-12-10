# Tratamiento de textos
# -----------------------------------------------------------------------
from collections import Counter

# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np
import pickle
import joblib

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import plotly_express as px
from IPython.display import display, HTML
from wordcloud import WordCloud

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


# NLP / SISTEMAS RECOMENDACION
# -----------------------------------------------------------------------
import spacy
from nltk.corpus import stopwords
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

# Variables Globales
# -----------------------------------------------------------------------
DATOS_RUTA_BASE = os.path.join(PROJECT_ROOT, 'datos')
DATOS_RUTA_TRATADOS = os.path.join(DATOS_RUTA_BASE, 'tratados')
DATOS_RUTA_CLUSTERS = os.path.join(DATOS_RUTA_TRATADOS, 'clusters')
TRANSFORMERS_RUTA_BASE = os.path.join(PROJECT_ROOT, 'transformers/base')
TRANSFORMERS_RUTA_GOOD_ONES = os.path.join(PROJECT_ROOT, 'transformers/good_ones')



# importar funciones de soporte
# -----------------------------------------------------------------------
from src.eda import soporte_preprocesamiento as sup_prep
from src.eda import soporte_nulos as sup_nul
from src.eda import soporte_outliers as sup_out
from src.clasificacion import soportefeaturescaling as sup_fea
from src.eda import soporte_encoding2 as sup_encod
from src.clasificacion import soporte_modelos_clasificacion as sup_models
from src.clustering import soporte_clustering as sup_clus
from src.clustering import soporte_sarima as sup_sarimas
from src.eda import soporte_series_temporales as sup_series_tem
from src.regresion import soporte_regresion as sup_regre
from src.recomendacion import soporte_sistemas_recomendacion as sup_reco
from src.nlp import soporte_nlp as sup_nlp

##aplicar configuraciones
#------------------------------------------------------------------------
warnings.filterwarnings('ignore')
pd.set_option('display.max_info_columns', 50)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format) #eliminamos la notacion cientifica

tqdm.pandas()