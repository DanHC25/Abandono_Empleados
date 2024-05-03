# Cargar librerías
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


## Generación de Insights ##

# Cuantificación del problema: ¿Cual es la tasa de abandono?
df["abandono"].value_counts(normalize=True) * 100

# Transformar abandono a numérica
df['abandono'] = df.abandono.map({'No':0, 'Yes':1})

# Análisis de abandono por educación
temp = df.groupby('educacion')["abandono"].mean().sort_values(ascending = False) * 100
temp.plot.bar()
plt.show()

# Análisis de abandono por estado civil
temp = df.groupby('estado_civil')["abandono"].mean().sort_values(ascending = False) * 100
temp.plot.bar()
plt.show()

# Análisis de abandono por horas extras
temp = df.groupby('horas_extra')["abandono"].mean().sort_values(ascending = False) * 100
temp.plot.bar()
plt.show()

# Análisis de abandono por puesto
temp = df.groupby('puesto')["abandono"].mean().sort_values(ascending = False) * 100
temp.plot.bar()
plt.show()

# Análisis de abandono por salario
temp = df.groupby('abandono')["salario_mes"].mean()
temp.plot.bar()
plt.show()

''' # Conclusiones:
    El perfil medio del empleado que deja la empresa es:
    1. Bajo nivel educativo
    2. Soltero
    3. Trabaja en ventas
    4. Bajo salario
    5. Alta carga de horas extras '''


## Impacto Económico de las Renuncias ##

''' Según el estudio "Cost of Turnover" del 'Center for American Progress':
    1. El coste de la fuga de los empleados que ganan menos de $30000 es del 16,1% de su salario
    2. El coste de la fuga de los empleados que ganan entre $30000-$50000 es del 19,7% de su salario
    3. El coste de la fuga de los empleados que ganan entre $50000-$75000 es del 20,4% de su salario
    4. El coste de la fuga de los empleados que ganan más de $75000 es del 21% de su salario '''

# Creamos una nueva variable salario_anual del empleado
df['salario_anual'] = df["salario_mes"].transform(lambda x: x*12)
df[['salario_mes','salario_anual']]

# Calculamos el impacto económico de cada empleado si deja la empresa
condiciones = [(df['salario_anual'] <= 30000),
               (df['salario_anual'] > 30000) & (df['salario_anual'] <= 50000),
               (df['salario_anual'] > 50000) & (df['salario_anual'] <= 75000),
               (df['salario_anual'] > 75000)]

#Lista de resultados
resultados = [df["salario_anual"] * 0.161, df["salario_anual"] * 0.197, df["salario_anual"] * 0.204, df["salario_anual"] * 0.21]

#Aplicamos select
df['impacto_abandono'] = np.select(condiciones,resultados, default = -999)

# ¿Cúanto nos ha costado este problema en el último año?
coste_total =  df.loc[df.abandono == 1].impacto_abandono.sum()
coste_total # Costo total de $2719005.912

# ¿Cuanto nos cuesta que los empleados no estén motivados? (pérdidas en implicación == Baja)
df.loc[(df.abandono == 1) & (df.implicacion == 'Baja')].impacto_abandono.sum()

# ¿Cuanto dinero podríamos ahorrar fidelizando mejor a nuestros empleados?
print(f"Reducir un 10% la fuga de empleados nos ahorraría {int(coste_total * 0.1)}$ cada año.") # $271900
print(f"Reducir un 20% la fuga de empleados nos ahorraría {int(coste_total * 0.2)}$ cada año.") # $543801
print(f"Reducir un 30% la fuga de empleados nos ahorraría {int(coste_total * 0.3)}$ cada año.") # $815701


## Modelo de Machine Learning ##

df_ml = df.copy()
df_ml.info()

## Preparación de los Datos para la Modelización ##

# Transformar todas las variables categóricas a númericas

from sklearn.preprocessing import OneHotEncoder

# Variables Categóricas
cat = df_ml.select_dtypes('O')

# Instanciamos
ohe = OneHotEncoder()

# Entrenamos
ohe.fit(cat)

# Aplicamos
cat_ohe = ohe.transform(cat)

# Convertimos la salida a una matriz densa
cat_ohe = cat_ohe.toarray()

# Ponemos los nombres a las columnas
cat_ohe = pd.DataFrame(cat_ohe, columns = ohe.get_feature_names_out(input_features = cat.columns)).reset_index(drop = True)

# Seleccionamos las variables numéricas para poder juntarlas a las cat_ohe
num = df.select_dtypes('number').reset_index(drop = True)

# Las juntamos todas en un dataframe final
df_ml = pd.concat([cat_ohe,num], axis = 1)
df_ml

