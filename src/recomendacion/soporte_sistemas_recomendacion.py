# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd

# Para visualización de datos
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt

def get_index_from_register(dataframe, column, name):
    
    return dataframe[dataframe[column] == name].index[0]


def get_name_from_index(dataframe, column, index):
    
    return dataframe[dataframe.index == index][column].values[0]



# def get_register_names(dataframe):
#     # y ahora buscamos el título
#     top_simiar_movies = {}
#     for i in peli_similares_ordenadas:
#         top_simiar_movies[sr.get_title_from_index(i[0], df_contenido)] = i[1]

#     # visualizamos los resultados
#     plt.figure(figsize=(10, 6))
#     sns.set_style("whitegrid")

#     # Crear gráfico de barras
#     sns.barplot(
#         x=list(top_simiar_movies.values()), 
#         y=list(top_simiar_movies.keys()), 
#         palette="mako"
#     )

#     # Añadir etiquetas y título
#     plt.title("Top Películas Similares Basado en Contenido", fontsize=16, pad=20)
#     plt.xlabel("Similitud", fontsize=12)
#     plt.ylabel("Películas", fontsize=12)

#     # Añadir valores al final de cada barra
#     for i, value in enumerate(top_simiar_movies.values()):
#         plt.text(value + 0.02, i, f"{value:.2f}", va='center', fontsize=10)

#     plt.tight_layout()



def plot(peli1, peli2, dataframe):
    """
    Genera un gráfico de dispersión que compara dos películas en un espacio de características.

    Parameters:
    ----------
    peli1 : str
        Nombre de la primera película a comparar.
    peli2 : str
        Nombre de la segunda película a comparar.
    dataframe : pd.DataFrame
        Un dataframe transpuesto donde las columnas representan películas y las filas características.

    Returns:
    -------
    None
        Muestra un gráfico de dispersión con anotaciones para cada película.
    """
    x = dataframe.T[peli1]     
    y = dataframe.T[peli2]

    n = list(dataframe.columns)    

    plt.figure(figsize=(10, 5))

    plt.scatter(x, y, s=0)      

    plt.title('Espacio para {} VS. {}'.format(peli1, peli2), fontsize=14)
    plt.xlabel(peli1, fontsize=14)
    plt.ylabel(peli2, fontsize=14)

    for i, e in enumerate(n):
        plt.annotate(e, (x[i], y[i]), fontsize=12)  

    plt.show();


def filter_data(df):
    """
    Filtra un dataframe de ratings basado en la frecuencia mínima de valoraciones por película y por usuario.

    Parameters:
    ----------
    df : pd.DataFrame
        Un dataframe con columnas 'movieId', 'userId' y 'rating'.

    Returns:
    -------
    pd.DataFrame
        Un dataframe filtrado que contiene solo las películas con al menos 300 valoraciones 
        y los usuarios con al menos 1500 valoraciones.
    """
    ## Ratings Per Movie
    ratings_per_movie = df.groupby('movieId')['rating'].count()
    ## Ratings By Each User
    ratings_per_user = df.groupby('userId')['rating'].count()

    ratings_per_movie_df = pd.DataFrame(ratings_per_movie)
    ratings_per_user_df = pd.DataFrame(ratings_per_user)

    filtered_ratings_per_movie_df = ratings_per_movie_df[ratings_per_movie_df.rating >= 300].index.tolist()
    filtered_ratings_per_user_df = ratings_per_user_df[ratings_per_user_df.rating >= 1500].index.tolist()
    
    df = df[df.movieId.isin(filtered_ratings_per_movie_df)]
    df = df[df.userId.isin(filtered_ratings_per_user_df)]
    return df