


import itertools
from scipy.special import factorial, comb, perm
from collections import Counter

def permutaciones(elementos):
    """
    Genera todas las permutaciones de los elementos proporcionados y cuenta el número de permutaciones.
    args: 
        elementos: lista de elementos a permutar
    returns:    
        permutaciones_list: lista de todas las permutaciones
        num_permutaciones: número de permutaciones

    Ejemplo:
    elementos = ['A', 'B', 'C']
    permutaciones_list, num_permutaciones = permutaciones(elementos)
    print(f"El número de permutaciones de los elementos {elementos} es: {num_permutaciones}")
    print("Las permutaciones son:")
    for p in permutaciones_list:
        print(p)

    Salida:
    El número de permutaciones de los elementos ['A', 'B', 'C'] es: 6
    Las permutaciones son:
    ('A', 'B', 'C')
    ('A', 'C', 'B')
    ('B', 'A', 'C')
    ('B', 'C', 'A')
    ('C', 'A', 'B')
    ('C', 'B', 'A')

    """
    n = len(elementos)  # Número de elementos
    # Generar todas las permutaciones usando itertools.permutations
    permutaciones_list = list(itertools.permutations(elementos))
    
    # Contar las permutaciones usando scipy.special.factorial
    num_permutaciones = factorial(n, exact=True)
    
    return permutaciones_list, num_permutaciones



def variaciones(elementos, r):
    """
    Genera todas las variaciones de los elementos tomados de r en r y cuenta el número de variaciones.
    args: 
        elementos: lista de elementos a variar
        r: número de elementos a variar
    returns:

        variaciones_list: lista de todas las variaciones
        num_variaciones: número de variaciones

    Ejemplo:
    elementos = ['A', 'B', 'C']
    r = 2
    variaciones_list, num_variaciones = variaciones(elementos, r)
    print(f"El número de variaciones de los elementos {elementos} tomados de {r} en {r} es: {num_variaciones}")
    print("Las variaciones son:")
    for v in variaciones_list:
        print(v)

    Salida:
    El número de variaciones de los elementos ['A', 'B', 'C'] tomados de 2 en 2 es: 6
    Las variaciones son:
    ('A', 'B')
    ('A', 'C')
    ('B', 'A')
    ('B', 'C')
    ('C', 'A')
    ('C', 'B')
    """
    n = len(elementos)  # Número total de elementos
    # Generar todas las variaciones usando itertools.permutations
    variaciones_list = list(itertools.permutations(elementos, r))
    
    # Contar las variaciones usando scipy.special.perm
    num_variaciones = perm(n, r, exact=True)
    
    return variaciones_list, num_variaciones


def combinaciones(elementos, r):
    """
    Genera todas las combinaciones de los elementos tomados de r en r y cuenta el número de combinaciones.
    args: 
        elementos: lista de elementos a combinar
        r: número de elementos a combinar
    returns:
        combinaciones_list: lista de todas las combinaciones
        num_combinaciones: número de combinaciones

    Ejemplo:
    elementos = ['A', 'B', 'C']
    r = 2
    combinaciones_list, num_combinaciones = combinaciones(elementos, r)
    print(f"El número de todas las agrupaciones posibles que pueden hacerse con los {len(elementos)} elementos de forma que se tomen {r} en cada combinación es: {num_combinaciones}")
    print("Las combinaciones son:")
    for c in combinaciones_list:
        print(c)

    Salida:
    El número de todas las agrupaciones posibles que pueden hacerse con los 3 elementos de forma que se tomen 2 en cada combinación es: 3
    Las combinaciones son:
    ('A', 'B')
    ('A', 'C')
    ('B', 'C')
    """
    n = len(elementos)  # Número total de elementos
    # Generar todas las combinaciones usando itertools.combinations
    combinaciones_list = list(itertools.combinations(elementos, r))
    
    # Contar las combinaciones usando scipy.special.comb
    num_combinaciones = comb(n, r, exact=True)
    
    return combinaciones_list, num_combinaciones


def permutaciones_con_repeticion(elementos):
    """
    Genera todas las permutaciones con repetición de los elementos y cuenta el número de permutaciones.
    args: 
        elementos: lista de elementos a permutar
    returns:
        permutaciones_list: lista de todas las permutaciones
        num_permutaciones: número de permutaciones

    Ejemplo:    
    elementos = ['A', 'A', 'B']
    permutaciones_list, num_permutaciones = permutaciones_con_repeticion(elementos)
    print(f"El número de permutaciones con repetición de los elementos {elementos} es: {num_permutaciones}")
    print("Las permutaciones son:")
    for p in permutaciones_list:
        print(p)

    Salida:
    El número de permutaciones con repetición de los elementos ['A', 'A', 'B'] es: 3
    Las permutaciones son:
    ('A', 'A', 'B')
    ('A', 'B', 'A')
    ('B', 'A', 'A')
    """
    # Contar la frecuencia de cada elemento
    frec = Counter(elementos)
    
    # Calcular el número de permutaciones con repetición
    n = len(elementos)
    num_permutaciones = factorial(n, exact=True)
    for count in frec.values():
        num_permutaciones //= factorial(count, exact=True)
    
    # Generar todas las permutaciones posibles (considerando repetición)
    permutaciones_list = set(itertools.permutations(elementos))
    
    return list(permutaciones_list), num_permutaciones



def variaciones_con_repeticion(elementos, r):
    """
    Genera todas las variaciones con repetición de los elementos tomados de r en r y cuenta el número de variaciones.
    args: 
        elementos: lista de elementos a variar
        r: número de elementos a variar
    returns:
        variaciones_list: lista de todas las variaciones
        num_variaciones: número de variaciones

    Ejemplo:
    elementos = ['A', 'B', 'C']
    r = 2
    variaciones_list, num_variaciones = variaciones_con_repeticion(elementos, r)
    print(f"El número de variaciones con repetición de los elementos {elementos} tomados de {r} en {r} es: {num_variaciones}")
    print("Las variaciones son:")
    for v in variaciones_list:
        print(v)

    Salida:
    El número de variaciones con repetición de los elementos ['A', 'B', 'C'] tomados de 2 en 2 es: 9
    Las variaciones son:
    ('A', 'A')
    ('A', 'B')
    ('A', 'C')
    ('B', 'A')
    ('B', 'B')
    ('B', 'C')
    ('C', 'A')
    ('C', 'B')
    ('C', 'C')
    """
    # Generar todas las variaciones con repetición usando itertools.product
    variaciones_list = list(itertools.product(elementos, repeat=r))
    
    # Calcular el número de variaciones con repetición
    num_variaciones = len(elementos) ** r
    
    return variaciones_list, num_variaciones


def combinaciones_con_repeticion(elementos, r):
    """
    Genera todas las combinaciones con repetición de los elementos tomados de r en r y cuenta el número de combinaciones.
    args: 
        elementos: lista de elementos a combinar
        r: número de elementos a combinar
    returns:
        combinaciones_list: lista de todas las combinaciones
        num_combinaciones: número de combinaciones

    Ejemplo:
    elementos = ['A', 'B', 'C']
    r = 2
    combinaciones_list, num_combinaciones = combinaciones_con_repeticion(elementos, r)
    print(f"El número de combinaciones con repetición de los elementos {elementos} tomados de {r} en {r} es: {num_combinaciones}")
    print("Las combinaciones son:")
    for c in combinaciones_list:
        print(c)

    Salida:
    El número de combinaciones con repetición de los elementos ['A', 'B', 'C'] tomados de 2 en 2 es: 6
    Las combinaciones son:
    ('A', 'A')
    ('A', 'B')
    ('A', 'C')
    ('B', 'B')
    ('B', 'C')
    ('C', 'C')
    """
    n = len(elementos)  # Número total de elementos
    # Generar todas las combinaciones con repetición usando itertools.combinations_with_replacement
    combinaciones_list = list(itertools.combinations_with_replacement(elementos, r))
    
    # Calcular el número de combinaciones con repetición usando scipy.special.comb
    num_combinaciones = comb(n + r - 1, r, exact=True)
    
    return combinaciones_list, num_combinaciones



def producto_cartesiano(*conjuntos):
    """
    Genera todas las combinaciones posibles tomando un elemento de cada uno de los conjuntos
    y cuenta el número de combinaciones.
    args: 
        *conjuntos: lista de conjuntos
    returns:
        combinaciones_list: lista de todas las combinaciones
        num_combinaciones: número de combinaciones

    Ejemplo:
    conjunto1 = ['A', 'B', 'C']
    conjunto2 = ['x', 'y']
    conjuntos = [conjunto1, conjunto2]
    combinaciones_list, num_combinaciones = producto_cartesiano(*conjuntos)
    print(f"El número de combinaciones posibles tomando un elemento de cada uno de los {len(conjuntos)} conjuntos es: {num_combinaciones}")
    print("Las combinaciones son:")
    for c in combinaciones_list:
        print(c)

    Salida:
    El número de combinaciones posibles tomando un elemento de cada uno de los 2 conjuntos es: 6
    Las combinaciones son:
    ('A', 'x')
    ('A', 'y')
    ('B', 'x')
    ('B', 'y')
    ('C', 'x')
    ('C', 'y')

    """
    # Generar todas las combinaciones posibles usando itertools.product
    combinaciones_list = list(itertools.product(*conjuntos))
    
    # Calcular el número de combinaciones
    num_combinaciones = 1
    for conjunto in conjuntos:
        num_combinaciones *= len(conjunto)
    
    return combinaciones_list, num_combinaciones
