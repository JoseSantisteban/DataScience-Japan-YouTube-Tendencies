<div style="width: 100%; clear: both;">
<div style="float: left; width: 50%;">
<img src="https://estudiaperu.pe/wp-content/uploads/2021/04/UPC-carreras-para-gente-que-trabaja.png", align="left" style="height: 150px; width:300px;>
</div>

<div style="float: right; width: 50%;">
<p style="margin: 0; padding-top: 22px; text-align:right;">Curso: Administración de la información. </p>
<p style="margin: 0; text-align:right;">2022 · Carrera de Ciencias de la Computación. Trabajo Final del Curso</p>
<p style="margin: 0; text-align:right; padding-button: 100px;">Alumno: <b>José Mauricio Santisteban Cerna</b> - <a href="">josesantisteban062@gmail.com</a></p>
</div>
</div>
<div style="width:100%;">&nbsp;</div>
<center><h1>Data Science: Análisis de la tendencia de los videos de Youtube en Japón</h1></center>

# Contenidos

  1. [Objetivos del proyecto ] (#data1)
  2. [Caso de Análisis ](#data2)
  3. [Conjunto de Datos ](#data3)
  4. [Análisis de los datos ](#data4)
  5. [Código ](#data5)
  6. [Conclusiones ](#data6)

## 1. Objetivos del proyecto <a name="data1"></a>
El presente trabajo detalla el desarrollo de un proyecto de analítica mediante un análisis exploratorio de datos a fin de crear nuevo conocimiento y resolver problemas de modelación de datos. Para ello, se ha hecho uso de un conjunto de datos sin procesar sobre estadísticas de videos de la plataforma de streaming YouTube Japón cuya información servirá para hallar respuesta a interrogantes propuestas.

En cuanto a información de los videos:

* ¿Qué categorías de videos son las de mayor tendencia? 
* ¿Qué categorías de videos son los que más gustan? ¿Y las que menos gustan? 
* ¿Qué categorías de videos tienen la mejor proporción (ratio) de “Me gusta” / “No me gusta”? 
* ¿Qué categorías de videos tienen la mejor proporción (ratio) de “Vistas” / “Comentarios”? 
* ¿Cómo ha cambiado el volumen de los videos en tendencia a lo largo del tiempo? 
* ¿Qué canales de YouTube son tendencia más frecuentemente?  ¿Y cuáles con menos frecuencia?
* ¿En qué Estados se presenta el mayor número de “Vistas”, “Me gusta” y “No me gusta”? 

Sobre posible creación de modelos de predicción de datos:

* ¿Es factible predecir el número de “Vistas” o “Me gusta” o “No me gusta”? 
* ¿Los videos en tendencia son los que mayor cantidad de comentarios positivos reciben?

## 2. Caso de Análisis <a name="data2"></a>
El conjunto de datos (dataset) a utilizar fue alterado para fines del ejercicio analítico, pero el dataset original proviene del sitio web [Kagle](https://www.kaggle.com/). . Además, de acuerdo a su descripción presente en la página web, los datos fueron recolectados por el colaborador Mitchell J, dueño del dataset, con última actualización el 02 de junio de 2019 bajo el nombre de [‘Trending YouTube Video Statistics: Daily statistics for trending YouTube Videos’](https://www.kaggle.com/datasets/datasnaek/youtube-new). 

Los datos fueron obtenidos usando directamente la API de YouTube e incluye meses de recolección de data sobre los videos más virales de la plataforma divididos en distintos países con más de 200 entradas por día. Cada conjunto está separado por país y dividido en su respectivo archivo csv y json, conteniendo la información por video y las descripciones de las categorías únicas de video respectivamente.

La recolección de este tipo de datos pueden ser muy útiles para estudios de caso aplicables. Por primera instancia, el análisis de la data puede contribuir a la creación de información que puede ser relevante para la plataforma. Mientras más datos sean analizados en el tiempo, se pueden crear modelos de predicción sobre qué factores pueden afectar la popularidad de un tipo específico de videos.


## 3. Conjunto de Datos <a name="data3"></a>
Los archivos a utilizar para el análisis de datos corresponden a los videos virales en la región de Japón, por ende los archivos utilizados son los nombrados: JPvideos_cc50.csv y JP_category_id.json.

Para fines del trabajo, el archivo csv fue alterado agregándole 4 nuevas columnas (state, lat, lon, geometry). Por ello, el dataset se comprende la siguiente información descrita:

* video_id: identificador individual por video viral del país.
* trending_date: fecha donde se presenció viralidad en la plataforma.
* title: titulo del video en idioma original.
* channel_title: título del canal que publicó el video viral en idioma original.
* category_id: id de la categoría del video viral.
* publish_time: timestamp de la publicación original del video.
* tags: etiquetas presentes en el video dentro de la plataforma.
* views: cantidad de vistas del video a la fecha.
* likes: número de ‘me gusta’ del video hasta la fecha.
* dislikes: número de ‘no me gusta’ del video hasta la fecha.
* comment_count: número de comentarios registrados en el video hasta la fecha.
* thumbnail_link: link de la fotografía principal del video publicado.
* comments_disabled: bool sobre la configuración de los comentarios del video, activados o desactivados.
* ratings_disabled: bool sobre la configuración de los rating, calificaciones, del video publicado, activado o desactivado.
* video_error_or_removed: bool sobre el estado del video en la plataforma en la actualidad, si hay error al reproducirlo o fue removido.
* description: descripción en idioma original presente en el video publicado.
* state: nombre del Estado perteneciente al país.
* lat: latitud geográfica de ubicación del Estado. 
* lon: longitud geográfica de ubicación del Estado. 
* geometry: registra las coordenadas de las geometrías donde su ubica el Estado dentro del planeta.

## 4. Análisis de los datos <a name="data4"></a>
Primeramente, importamos las librerías para observar y manipular los datos dentro del csv y json.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import datetime
import json
%matplotlib inline
import geopandas 
```
Creamos la variable con el dataset original llamado JPvideos y observamos una visualización previa de su contenido.
````python
JPvideos = pd.read_csv('JPvideos_cc50.csv')
JPvideos.head()
````
<table>
<thead><tr><th scope=col>video_id</th><th scope=col>trending_date</th><th scope=col>title</th><th scope=col>channel_title</th><th scope=col>category_id</th><th scope=col>publish_time</th><th scope=col>tags</th><th scope=col>views</th><th scope=col>likes</th><th scope=col>dislikes</th><th scope=col>comment_count</th><th scope=col>thumbnail_link</th><th scope=col>comments_disabled</th><th scope=col>ratings_disabled</th><th scope=col>video_error_or_removed</th><th scope=col>description</th><th scope=col>state</th><th scope=col>lat</th><th scope=col>lon</th><th scope=col>geometry</th></tr></thead>
<tbody>
	<tr><td>5ugKfHgsmYw</td><td>18/07/2002</td><td>陸自ヘリ、垂直に落下＝路上の車が撮影</td><td>時事通信映像センター</td><td>25.0</td><td>05/02/2018 22:04</td><td>事故|"佐賀"|"佐賀県"|"ヘリコプター"|"ヘリ"|"自衛隊"|"墜落"|"落下"|"現...</td><td>188085.0</td><td>591.0</td><td>189.0</td><td>0.0</td><td>https://i.ytimg.com/vi/5ugKfHgsmYw/default.jpg </td><td>VERDADERO</td><td>FALSO</td><td>FALSO</td><td>佐賀県神埼市の民家に墜落した陸上自衛隊のＡＨ６４Ｄ戦闘ヘリコプターが垂直に落下する様子を、近...</td><td>Kyoto</td><td>35.450406</td><td>135.333331</td><td>POINT (135.3333309 35.4504059)</td></tr>
</tbody>
</table> 
Con la función describe() se visualiza la distribución de las variables numéricas dentro del dataset original, así como la media, la media, la desviación estándar y los percentiles.
Mientras que con la función info() observamos la distribución de los tipos de datos que contiene cada columna, así como la cantidad total de entradas y el número de datos no vacíos. Este último servirá para el preprocesamiento de los datos.
````python 
JPvideos.info()
#las columnas con más datos faltantes son 'trending_date' y 'description'
#hay varias columnas con cerca de 1.2k datos faltantes que pueden ser obviados o eliminados
````
````python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 21718 entries, 0 to 21717
Data columns (total 20 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   video_id                21445 non-null  object 
 1   trending_date           8318 non-null   object 
 2   title                   20528 non-null  object 
 3   channel_title           20528 non-null  object 
 4   category_id             20524 non-null  float64
 5   publish_time            20523 non-null  object 
 6   tags                    20525 non-null  object 
 7   views                   20522 non-null  float64
 8   likes                   20522 non-null  float64
 9   dislikes                20522 non-null  float64
 10  comment_count           20522 non-null  float64
 11  thumbnail_link          20522 non-null  object 
 12  comments_disabled       20522 non-null  object 
 13  ratings_disabled        20522 non-null  object 
 14  video_error_or_removed  20522 non-null  object 
 15  description             18398 non-null  object 
 16  state                   21718 non-null  object 
 17  lat                     21718 non-null  float64
 18  lon                     21718 non-null  float64
 19  geometry                21718 non-null  object 
dtypes: float64(7), object(13)
memory usage: 3.3+ MB
````
Observamos la distribución de las columnas para poder referenciarlas en el posterior tratamiento y limpieza de datos
````python
JPvideos.columns
````
````python
Index(['video_id', 'trending_date', 'title', 'channel_title', 'category_id',
       'publish_time', 'tags', 'views', 'likes', 'dislikes', 'comment_count',
       'thumbnail_link', 'comments_disabled', 'ratings_disabled',
       'video_error_or_removed', 'description', 'state', 'lat', 'lon',
       'geometry'],
      dtype='object')
````
Asimismo, con la función pairplot de la libreria seaborn podemos tener un vistazo inicial a la correlación entre las distintas columnas de datos numéricos. En primera instancia, no se observa relación directa entre ninguna de las variables entre sí que sean relevantes para el análisis solicitado.
````python
sns.pairplot(JPvideos)
````
<img src="/img/pairplot.png" style="height: 800px; width:700px;"/>

Con la comprensión de los datos observados en la sección anterior, podemos proceder a la fase de preprocesamiento de los mismos. 

Para comenzar, se procedió a eliminar las columnas cuya información no sería relevante para hallar solución a las interrogantes propuestas. Haciendo copia del dataset en la nueva variable new_df se eliminaron las columnas irrelevantes. Así, permanecen las columnas: channel_title, category_id, publish_time, views, likes, dislikes, comment_count, state, lat, lon, y geometry.

````python
new_df = JPvideos
new_df = new_df.drop('comments_disabled', axis=1)
new_df = new_df.drop('description', axis=1)
new_df = new_df.drop('ratings_disabled', axis=1)
new_df = new_df.drop('video_error_or_removed', axis=1)
new_df = new_df.drop('thumbnail_link', axis=1)
new_df = new_df.drop('tags', axis=1)
new_df = new_df.drop('trending_date', axis=1)
new_df = new_df.drop('title', axis=1)
new_df = new_df.drop('video_id', axis=1)
````
Como observamos en la inspección de los datos, hay varias columnas que contienen datos vaciós NaN; por lo que procedemos a imprimir la cantidad de datos faltantes en el nuevo dataset.

````python
print(new_df.isna().sum())
````
````python
channel_title    1190
category_id      1194
publish_time     1195
views            1196
likes            1196
dislikes         1196
comment_count    1196
state               0
lat                 0
lon                 0
geometry            0
dtype: int64
````
Procedemos a eliminar los datos faltantes. Hemos considerado esta opción puesto que los datos faltantes son de importancia alta para el análisis y no se pueden sacar en base al dataset como por ejemplo el nombre del canal, categoría, etc. 
Visualizamos la nueva cantidad de datos faltantes y corroboramos que en efecto el dataset está completo.
````python
#Limpieza de datos
new_df.dropna(subset=["views"], inplace=True)
#Eliminamos valores NaN del dataset
new_df=new_df.dropna()
print(new_df.isna().sum())
````
````python
channel_title    0
category_id      0
publish_time     0
views            0
likes            0
dislikes         0
comment_count    0
state            0
lat              0
lon              0
geometry         0
dtype: int64
````
Para facilitar las operaciones en la parte de visualización, se convierten las columnas de datos implicadas en tipo entero, float o datetime timestamp correspondientemente. Seguidamente leemos el archivo json para obtener los nombres de las categorías de los videos en el nuevo dataset. El diccionario se guarda en la variable categorynames.
````python
#datos a int
new_df['category_id'] = new_df['category_id'].astype('int')
new_df['likes'] = new_df['likes'].astype(float)
new_df['views'] = new_df['views'].astype(float)
new_df['dislikes'] = new_df['dislikes'].astype(float)
new_df['publish_time'] = pd.to_datetime( new_df['publish_time']) 

#leer json

f = open('JP_category_id.json')

datajson = json.load(f)
datajson

categorynames = {}


for item in datajson['items']:
  categorynames[ int(item['id'])] = item['snippet']['title']

print(categorynames)
````
Debido a la gran cantidad de datos a analizar, cerca de 20 mil, debemos cerciorarnos de la distribución de estos. Por lo que procedemos a inspeccionar a mayor detalle los datos de las columnas dislikes, likes, views y comment_count en busca de datos atípicos que alterarían significativamente el análisis, con el fin de obtener resultados más precisos y reales. Utilizamos la función boxplot, el cual es un método de la librería seaborn para graficar los datos en cuartiles. Debería poder observarse el rango de los datos (whiskers) el primer y tercer cuartil junto con la mediana en la caja del gráfico, junto con unos cuantos datos atípicos.

Datos atípicos en dislikes relacionándolos con la columna category_id: 
````python
#Buscamos datos atipicos
plt.figure(figsize=(15, 15))
sns.boxplot(x='category_id',y='dislikes',data=new_df,palette='winter')
````
<img src="/img/atypic.png" style="height: 800px; width:900px;"/>
De esta gráfica podemos observar la enorme cantidad de datos anómalos presentes en los datos necesarios para el análisis. No se puede observar ni el rango de los datos ni datos vitales como la mediana de los datos. Por ello, procedemos a cambiar los datos anómalos mediante la técnica de flooring y capping. Esta es una técnica de tratamiento para datos atípicos, a través de la cual se reemplazan los datos atípicos que exceden un máximo teórico por este valor (capping) o cambiando los datos menores a un mínimo teórico por este valor (flooring), utilizando los cuartiles reales de cada conjunto de datos.

Reemplazamos outliers en dislikes e imprimimos los nuevos datos.
````python
#Para dislikes
dislikes_tenth_p = np.percentile(new_df['dislikes'], 10)
dislikes_ninetieth_p = np.percentile(new_df['dislikes'], 90)
new_df['dislikes'] = np.where(new_df['dislikes']<dislikes_tenth_p, dislikes_tenth_p, new_df['dislikes'])
new_df['dislikes'] = np.where(new_df['dislikes']>dislikes_ninetieth_p, dislikes_ninetieth_p,new_df['dislikes'])
new_df['dislikes']`
````
Después de reemplazar los datos atípicos procedemos a verificar la nueva distribución de cada conjunto de variables.

Boxplot de dislikes vs category_id con menos outliers:
````python
plt.figure(figsize=(15, 15))
sns.boxplot(x='category_id',y='dislikes',data=new_df,palette='winter')
````
<img
src="/img/boxfigure.png" style="height: 800px; width:900px;"/>

Ahora que los datos están completos y mejor distribuídos, se pueden procesar para resolver las dudas de no modelamiento planteadas.

**Pregunta 1: ¿Qué categorías de videos son las de mayor tendencia?**
<img
src="/img/p1.png" style="height: 800px; width:900px;"/>
De acuerdo con el gráfico, determinamos que las categorías con más videos en tendencia son: Entretenimiento, Personas y blogs, Deportes, Noticias y políticas, y Música.

**Pregunta 2: ¿Qué categorías de videos son los que más gustan? ¿Y las que menos gustan?**
<img
src="/img/p2.png" style="height: 900px; width:900px;"/>
De acuerdo con las gráficas, los videos que tienen mayor cantidad de likes son: Entretenimiento, Música y Personas y blogs.
Y los que tienen mayor cantidad de dislikes son: Entretenimiento, Personas y blogs, y Deportes.

**Pregunta 3:¿Qué categorías de videos tienen la mejor proporción (ratio) de “Me gusta” / “No me gusta”?**
<img
src="/img/p3.png" style="height: 500px; width:900px;"/>
De acuerdo con la gráfica, la mejor categoría con mejor proporción entre likes y dislikes es Mascotas y animales, con más de 30 likes por cada dislike. Después le siguen Música, Films y animación, y Gaming.

**Pregunta 4:¿Qué categorías de videos tienen la mejor proporción (ratio) de “Vistas” / “Comentarios”?**
<img
src="/img/p4.png" style="height: 500px; width:900px;"/>
De acuerdo con la gráfica, la mejor categoría con mejor proporción entre vistas y comentarios es Deportes, con más de 500 vistas por cada comentario. Después le siguen Autos y vehículos, Films y animación.


**Pregunta 5:¿Cómo ha cambiado el volumen de los videos en tendencia a lo largo del tiempo?**
<img
src="/img/p5.png" style="height: 500px; width:900px;"/>
De acuerdo con la gráfica, se puede ver que hay más variedad de videos en tendencia entre los meses de febrero hasta mayo.


**Pregunta 6: ¿Qué canales de YouTube son tendencia más frecuentemente? ¿Y cuáles con menos frecuencia?**
<img
src="/img/p6.png" style="height: 500px; width:900px;"/>
De acuerdo con nuestra información, podemos observar que los 5 primeros canales sobrepasan los 80 videos en tendencia. Donde el primer canal es un canal de noticias y el resto son creadores de contenido independientes.
En el caso de los últimos canales, más de mil canales tienen solo un video, por lo que en la gráfica solo pusimos 4 canales aleatorios.


**Pregunta 7: ¿En qué Estados se presenta el mayor número de “Vistas”, “Me gusta” y “No me gusta”?**

Para mejor procesamiento de los datos para graficación geográfica se usó la librería geopandas y japanmap para ilustrar mejor los resultados obtenidos.
````python
gdf = geopandas.GeoDataFrame(
    new_df, geometry=geopandas.points_from_xy(new_df.lon, new_df.lat))
````

Con los puntos geométricos se pueden visualizar las cantidades solicitadas por el total de estados (prefecturas) de la ciudad el sol naciente.
````python
fig, ax = plt.subplots(figsize=(10,12))
gdf.plot(ax=ax, column='views', cmap='coolwarm', legend=True, legend_kwds={'shrink':0.3})
ax.axis('on');
ax.set_title('views per State', fontdict={'fontsize':'25','fontweight':'3'})
````
<img
src="/img/views.png" style="height: 500px; width:900px;"/>
Likes por prefectura
````python
fig, ax = plt.subplots(figsize=(10,12))
gdf.plot(ax=ax, column='likes', cmap='inferno', legend=True, legend_kwds={'shrink':0.3})
ax.axis('on');
ax.set_title('likes per State', fontdict={'fontsize':'25','fontweight':'3'})
````
<img
src="/img/likes.png" style="height: 500px; width:900px;"/>
Dislikes por prefectura
````python
fig, ax = plt.subplots(figsize=(10,12))
gdf.plot(ax=ax, column='dislikes', cmap='magma', legend=True, legend_kwds={'shrink':0.3})
ax.axis('on');
ax.set_title('dislikes per State', fontdict={'fontsize':'25','fontweight':'3'})
````
<img
src="/img/dislikes.png" style="height: 500px; width:900px;"/>
En cuanto a la distribución de las vistas, es observable que la tendencia general es que la mayoría de los videos provenientes de ciertos estados no sobrepasan las 200 mil vistas, ubicándose las más altas en las prefecturas ubicadas en el sur: podría estimarse que se trata de Kochi y Mie (hasta 500 mil vistas).
Con respecto a los likes, la distribución representa que hay poca recepción positiva de los videos provenientes de la mayoría de prefecturas, pues no sobrepasan los 2000 ‘me gusta’. Los valores máximos se localizan en Aomori, Akita y Aichi (más de 7000).
En referente a los dislikes, la distribución es similar para casi todas las prefecturas, entre 200 y 100 ‘no me gusta’. Hay un máximo de 500 dislikes en Kochi, y una de 400 en Mie. Aunque los videos en general no reciban tantos likes, la cantidad de ‘no me gusta’ se vuelve insignificante en comparación, resultando que se pueda inferir que los videos provenientes de japón tienen una acogida y aprobación apropiada.

Finalmente, guardamos el nuevo dataset generado:
````python
df.to_csv('new_df.csv', index=False)
````
## 5. Código <a name="data5"></a>
Como he optado por hacer directamente en un Jupyter notebook, todos las partes del código estan presentes en este documento, sin embargo, en el repositorio de Github también es posible encontrar todos los archivos utilizados durante la práctica.



## 6. Conclusiones <a name="data6"></a>
Las conclusiones resultan de las respuestas que cada equipo proporcionara por cada uno de los requerimientos del proyecto. 

* Pregunta 1:
Las tres primeras categorías abarcan más del 50% del total de videos

* Pregunta 2:
La proporción de dislikes en comparación a los likes en todos los videos en tendencia es mínima, concluyendo que no se hace uso frecuente de esta característica de YouTube en Japón.

* Pregunta 3:
En todas las categorías predomina la cantidad de likes sobre los dislikes.

* Pregunta 4:
La mayoría de categorías posee una proporción similar entre comentarios y vistas, sin embargo, la categoría de música es la que posee mayor cantidad de comentarios.

* Pregunta 6:
Solo hay un 10% de canales que destacan en tendencia, el resto solo son canales que tienen pocos videos en tendencia, siendo casi el 90% del total de canales.

* Pregunta 7:
Son muy pocos los estados de Japón que sobresalen en cantidad de ‘me gusta’, ‘no me gusta’ y visitas por video. Aún así, siguen una tendencia clara. Las prefecturas que responden a la incógnita son Kochi, Mie, Aomori, Akita y Aichi. Estas pueden servir para hacer pruebas de predicción para crear modelos de regresión.


