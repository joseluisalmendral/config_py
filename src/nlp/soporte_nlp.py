import re
import string
import contractions
import pandas as pd


def eliminar_emojis(text):
    """
    Elimina emojis de una cadena de texto.

    Parámetros:
        text (str): El texto del cual se eliminarán los emojis.

    Retorna:
        str: El texto sin emojis.
    """
    # Expresión regular que elimina los emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticones
        "\U0001F300-\U0001F5FF"  # Símbolos y pictogramas
        "\U0001F680-\U0001F6FF"  # Transporte y mapas
        "\U0001F1E0-\U0001F1FF"  # Banderas (iOS)
        "\U00002702-\U000027B0"  # Símbolos adicionales
        "\U000024C2-\U0001F251"  # Otros caracteres especiales
        "]+", flags=re.UNICODE
    )

    return emoji_pattern.sub(r'', text)



def limpiar_texto(texto: str):
    """
    Limpia una cadena de texto aplicando varios preprocesamientos:
    - Convierte a minúsculas.
    - Elimina signos de puntuación.
    - Elimina emojis.
    - Elimina espacios al inicio y al final.
    - Elimina saltos de línea y múltiples espacios.
    - Elimina números.
    - Sustituye contracciones (especialmente en inglés).

    Parámetros:
        texto (str): El texto que se desea limpiar.

    Retorna:
        str: El texto limpio.
    """
    resultado = texto

    # Convertir a minúsculas
    resultado = resultado.lower()

    # Eliminar signos de puntuación
    translator = str.maketrans('', '', string.punctuation)
    resultado = resultado.translate(translator)

    # Eliminar emojis
    resultado = eliminar_emojis(resultado)

    # Eliminar espacios iniciales y finales
    resultado = resultado.strip()

    # Eliminar saltos de línea y múltiples espacios
    resultado = re.sub(r'\s+', ' ', resultado).strip()

    # Eliminar números
    resultado = re.sub(r'\d', ' ', resultado)

    # Sustituir contracciones
    resultado = contractions.fix(resultado)

    return resultado



def generar_columna_limpia(dataframe: pd.DataFrame, columna: str, nueva_columna: str):
    """
    Aplica la función 'limpiar_texto' a una columna específica de un DataFrame
    y guarda el resultado en una nueva columna.

    Parámetros:
        dataframe (pd.DataFrame): El DataFrame original.
        columna (str): El nombre de la columna que se desea limpiar.
        nueva_columna (str): El nombre de la nueva columna donde se guardará el texto limpio.

    Retorna:
        pd.DataFrame: El DataFrame con la nueva columna añadida.
    """
    dataframe[nueva_columna] = dataframe[columna].map(limpiar_texto)
    return dataframe