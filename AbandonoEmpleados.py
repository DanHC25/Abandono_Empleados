# Cargar librerías
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos de archivo csv
df = pd.read_csv('AbandonoEmpleados.csv', sep = ';', index_col= 'id', na_values='#N/D')

# Análisis de datos del archivo csv
df.info()

# Análisis de Datos Nulos
df.isna().sum().sort_values(ascending=False)
''' # Conclusiones:
    1. Columnas con valores nulos altos: años en puesto y Conciliación
    2. Análisis Exploratorio de Datos (EDA): Sexo, educación, satisfación de trabajo e implicación '''

# Eliminar columnas con valores nulos altos
df.drop(columns=["anos_en_puesto", "conciliacion"], inplace=True)

## Análisis Exploratorio de Datos (EDA): Graficos de Variables categóricas ##

# Seleccionamos las columnas categóricas del DataFrame
df_categorica = df.select_dtypes("O")
df_categorica # Se observa que son 14 columnas con variables categóricas

def graficos_eda_categoricos(cat):
    ''' Crear Gráficos de Variables Categóricas
        Arg 'cat': DataFrame con Varibles Categóricas'''

    filas = len(cat)//2 # Número de filas de la figura
    fig, ax = plt.subplots(nrows=filas, ncols=2, figsize=(16, filas*6)) # Creación de figura y subtramas

    for indice, valores in enumerate(cat):
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        # Calculamos las coordenadas del subgráfico
        fila = indice // 2  # División entera para obtener la posición de la fila del gráfico
        columna_grafico = indice % 2  # Módulo para obtener la posición de la columna del gráfico

        # Calculamos los conteos de valores únicos en la columna
        conteo = df_categorica[valores].value_counts()

        # Trazamos el gráfico de barras horizontal en el subgráfico correspondiente
        ax[fila, columna_grafico].barh(conteo.index, conteo.values, color='blue')
        ax[fila, columna_grafico].set_title(valores)

    plt.show()  # Mostramos la figura

graficos_eda_categoricos(df_categorica.columns[0:8])
graficos_eda_categoricos(df_categorica.columns[8:])

''' # Conclusiones:
    1. Columna 'mayor de edad' solo tiene un valor: Eliminarlo
    Sobre las imputaciones pendientes de variables categóricas:
    2. Columna educacion: imputar por 'Universitaria'
    3. Columna satisfaccion_trabajo: imputar por 'Alta'
    4. Columna implicacion: imputar por 'Alta' '''

df.drop(columns="mayor_edad", inplace=True) # Eliminar columna 'mayor de edad'
df["educacion"] = df["educacion"].fillna("Universitaria")
df["satisfaccion_trabajo"] = df["satisfaccion_trabajo"].fillna("Alta")
df["implicacion"] = df["implicacion"].fillna("Alta")

## Análisis Exploratorio de Datos (EDA): Variables Numéricas ##

def estadisticos_cont(num):
    ''' Describir las Variables Numéricas
        Arg 'num': DataFrame con Varibles Numéricas '''
    # Calculamos describe
    estadisticos = num.describe().T
    # Añadimos la mediana
    estadisticos['median'] = num.median()
    # Reordenamos para que la mediana esté al lado de la media
    estadisticos = estadisticos.iloc[:,[0,1,8,2,3,4,5,6,7]]
    return(estadisticos)

estadisticos_cont(df.select_dtypes("number"))

''' # Conclusiones:
    1. Columna 'empleados' solo tiene un valor --> Eliminarla
    2. Columna 'sexo' tiene 4 valores --> Eliminarla
    3. Columna 'horas quincena' solo tiene una valor --> Eliminarla
    4. De los nulos pendientes de imputación que sean numéricas solo está 'sexo', pero como la vamos a eliminar ya se imputa '''

df.drop(columns=["empleados", "sexo", "horas_quincena"], inplace=True)

