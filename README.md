<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>
# <h1 align=center>PROYECTO INDIVIDUAL Nº1</h1>
<h2 style="color:red"><strong><center>Machine Learning Operations (MLOps)<center></strong></h1>

<p>El siguiente proyecto soluciona un problema de negocio usando ML, el cual tiene como objetivo crear un sistema de 
recomendación de una plataforma de streaming (series y películas). Se comienza haciendo un trabajo de Data Engineer 
(ETL) debido a que los datos están anidados y sin transformar al momento de realizar la ingesta de datos de los 2 archivos 
que contienen dichos datos, los cuales son: movies_dataset.csv y credits.csv. Luego de realizado el ETL se crea el dataset movies_credits.csv.</p>

<hr style="color: Black">
<p>Ya que los datos están limpios en el nuevo dataset, se realiza un Análisis exploratorio de los datos (Exploratory Data Analysis-EDA), 
    lo que nos permite investigar las relaciones que hay entre las variables del dataset, 
    ver si hay algún patrón interesante que valga la pena explorar en un análisis posterior. Por lo tanto, un mapa de calor y 
    un gráfico de cajas permite conocer qué tantos registros nulos y outliers o anomalías existen. Asímismo, las nubes de palabras dan 
    una buena idea de cuáles palabras son más frecuentes en las columnas, las cuales podrían ayudar al sistema de recomendación.</p>

<hr style="color: Black">
<p>Por otra parte el datset movies_credits.csv alimenta una API en donde se disponibilizan los datos de la empresa. 
    Esta API se creó usando el framework FastAPI. En ella se pueden realizar consultas, creadas a partir de 6 funciones 
    (endpoints que se consumirán en la API):</p>

<ul>
<li> <mark>def peliculas_idioma( Idioma: str ):</mark> Se ingresa el idioma, y se retorna la cantidad de peliculas producidas en el mismo.
    Se valida el ingreso del idioma en mayúsculas dado que en el dataset aparecen en minúsculas.
    Si un idioma no está dentro del dataset este dará como resultado 0.</li>

<li> <mark>def peliculas_duracion( Pelicula: str ):</mark> Se ingresa el nombre de la pelicula, retornando la duracion y el año de la misma en formato lista dado que pueden
    haber películas que fueron lanzadas con el mismo nombre pero que son totalmente diferentes en cuanto a contenido (Ej: 'Heat').
    Si el nombre de la pelicula no está dentro del dataset este dará como resultado una lista vacía.</li>

<li> <mark>def franquicia( Franquicia: str ):</mark> Se ingresa el nombre de la franquicia, retornando la cantidad de peliculas, ganancia total y promedio.
    Se valida el ingreso de una franquicia que no está dentro del dataset como: La franquicia no existe.</li>

<li> <mark>def peliculas_pais( Pais: str ):</mark> Se ingresa el pais, retornando la cantidad de peliculas producidas en el mismo.
    Se valida el ingreso del idioma en minúsculas dado que en el dataset aparecen en mayúsculas.
    Si el nombre de un país no está dentro del dataset este dará como resultado 0.</li>

<li> <mark>def productoras_exitosas( Productora: str ):</mark> Se ingresas el nombre de la productora, entregando el revunue total y la cantidad de peliculas que realizó.
    Si el nombre de la productora no está dentro del dataset este dará como resultado 0 en los campos revenue y cantidad.</li>

<li> <mark>def get_director( nombre_director ):</mark> Se ingresa el nombre de un director que se encuentre dentro del dataset devolviendo el éxito del mismo medido a través del retorno,
    el cual se calculó como: (revenue_total - budget_total) / budget_total. 
    Además, devuelve el nombre de cada película, la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.
    Todas las respuestas están en formato lista.</mark></li>
</ul>

<hr style="color: Black">
<p>Una vez que toda la data es consumible por la API y el EDA permite entender bien los datos a los que tenemos acceso, es hora de entrenar el 
modelo de machine learning para armar un sistema de recomendación de películas, el cual consiste en recomendar películas a los usuarios 
basándose en películas similares, por lo que se debe encontrar 
la similitud de puntuación entre esa película y el resto de películas, se ordenarán según el score de similaridad y devolverá una 
lista de Python con 5 valores, cada uno siendo el string del nombre de las películas con mayor puntaje, en orden descendente.</p>

<p>Para el sistema de recomendación se hace uso del método 'TF-IDF' (Frecuencia de Término – Frecuencia Inversa de Documento). 
    Este es un algoritmo muy común para transformar el texto en una representación significativa de números, es decir, 
    calcula la frecuencia con la que una palabra dada aparece dentro de un documento y le asigna una puntuación. Así que mediante 
    'TfidfVectorizer' se tokeniza dichas palabras, luego este aprende el vocabulario (palabras) y las ponderaciones inversas de frecuencia, 
    generadas con 'TfidfTransformer', lo que reduce la escala de las palabras que aparecen mucho.</p>

<p>Por otra parte, se hace uso de la columna 'name_belongs' del dataset para dar la recomendación de las películas. 
    Cabe aclarar que aunque es una de las columnas que más valores nulos tiene, es la que mejor resultado genera si se compara con las columnas 'genres', 
    'production_companies' y 'production_countries'.</p>
    
<p>También es importante aclarar el uso de los primeros 5000 registros del dataset para la recomendación puesto que si se usan más, 
    Render arroja error de memoria.</p>


<p>Este sistema de recomendacion se deploya como una función adicional de la API anterior llamada:</p>

<li> <mark>def recomendacion( titulo ):</mark> Se ingresa el nombre de una película y te recomienda las similares en una lista de 5 valores.
<hr style="color: Black">




<h3 style="color:red"><center>REPOSITORIO GITHUB</center></h3>

<p><mark>Repositorio Proyecto:</mark> <a>https://github.com/aldemarbr94/PI_MLOps</a></p>

<p><mark>Dataset resultante del ETL (movies_credits.csv):</mark> https://raw.githubusercontent.com/aldemarbr94/PI_MLOps/main/movies_credits.csv</p>

<p>El <mark>ETL</mark> se encuentra en el archivo: <mark>PI_ML_OPS_Aldemar.ipynb</mark>.

<p>El <mark>EDA</mark> se encuentra en el archivo: <mark>PI_ML_OPS_Aldemar2.ipynb</mark>.

<p>Las <mark>funciones de la API</mark> se encuentran en el archivo: <mark>main.py</mark>.

<hr style="color: Black">




<h3 style="color:red"><center>RENDER</center></h3>

<p><mark>El servicio se encuentra en el siguiente enlace:</mark> <a>https://aldemarbr.onrender.com/docs</a>
