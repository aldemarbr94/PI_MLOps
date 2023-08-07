from fastapi import FastAPI
import pandas as pd
import numpy as np
import ast

url_csv = 'https://raw.githubusercontent.com/aldemarbr94/PI_MLOps/main/movies_credits.csv'
df_movies_credits = pd.read_csv(url_csv)

app = FastAPI()
@app.get("/")
def index():
    return 'PI_MLOps - Aldemar B'


@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma:str):
    '''Ingresas el idioma, retornando la cantidad de peliculas producidas en el mismo'''

    idioma = idioma.lower()
    cantidad_idioma = (df_movies_credits['original_language'][df_movies_credits['original_language']==idioma]).shape[0]

    return {'idioma':idioma, 'cantidad':cantidad_idioma}


    
@app.get('/peliculas_duracion/{pelicula}')
def peliculas_duracion(pelicula:str):
    '''Ingresas la pelicula, retornando la duracion y el año'''

    duracion = list(df_movies_credits['runtime'][df_movies_credits['title']==pelicula])
    anio = list(df_movies_credits['year_release_date'][df_movies_credits['title']==pelicula])

    return {'pelicula':pelicula, 'duracion':duracion, 'anio':anio}



@app.get('/franquicia/{franquicia}')
def franquicia(franquicia:str):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio'''

    cantidad_peliculasf = (df_movies_credits['title'][df_movies_credits['name_belongs']==franquicia]).shape[0]
    ganancia_t = (df_movies_credits['revenue']-df_movies_credits['budget'])[df_movies_credits['name_belongs']==franquicia].sum().round(2)
    ganancia_p = (df_movies_credits['revenue']-df_movies_credits['budget'])[df_movies_credits['name_belongs']==franquicia].mean().round(2)


    return {'franquicia':franquicia, 'cantidad':cantidad_peliculasf, 'ganancia_total':ganancia_t, 'ganancia_promedio':ganancia_p}



@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais:str):
    '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo'''

    pais = pais.upper()

    df = df_movies_credits['production_countries'].apply(ast.literal_eval)
    cantidad_peliculasp = 0

    for i in df:
        for n in i:
            if n == pais:
                cantidad_peliculasp += 1

    return {'pais':pais, 'cantidad':cantidad_peliculasp}



@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora:str):
    '''Ingresas la productora, entregandote el revunue total y la cantidad de peliculas que realizo '''

    cantidad = 0
    revenue_total = 0
    df = df_movies_credits['production_companies'].apply(ast.literal_eval)
    for i in range(len(df)):
        for n in df[i]:
            if n == productora:
                revenue_total += df_movies_credits['revenue'][i]
                cantidad += 1
    

    return {'productora':productora, 'revenue_total': revenue_total, 'cantidad':cantidad}




@app.get('/get_director/{nombre_director}')
def get_director(nombre_director:str):
    ''' Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. 
    Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma. En formato lista'''
    
    df = df_movies_credits['director'].apply(lambda x: "['Sin Director']" if type(x) != str else x).apply(ast.literal_eval)
    retorno_total_director = 0
    revenue_total = 0
    budget_total = 0
    lista_peliculas = []
    lista_anios = []
    lista_retorno_pelicula = []
    lista_budget_pelicula = []
    lista_revenue_pelicula = []

    for i in range(len(df)):
        for n in df[i]:
            if n == nombre_director:
                revenue_total += df_movies_credits['revenue'][i]
                budget_total += df_movies_credits['budget'][i]

                lista_peliculas.append(df_movies_credits['title'][i])
                lista_anios.append(df_movies_credits['year_release_date'][i].item())
                lista_retorno_pelicula.append(df_movies_credits['return'][i])
                lista_budget_pelicula.append(df_movies_credits['budget'][i])
                lista_revenue_pelicula.append(df_movies_credits['revenue'][i])

    if budget_total != 0:
        retorno_total_director += round((revenue_total - budget_total) / budget_total, 2)
    else:
        retorno_total_director

    return {'director':nombre_director, 'retorno_total_director':retorno_total_director, 
            'peliculas':lista_peliculas, 'anio':lista_anios, 'retorno_pelicula':lista_retorno_pelicula, 
            'budget_pelicula':lista_budget_pelicula, 'revenue_pelicula':lista_revenue_pelicula}




# ML
df_movies_credits = df_movies_credits.iloc[:,1:]

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
df_movies_credits['name_belongs'] = df_movies_credits['name_belongs'].fillna('')
tfidf_matrix = tfidf.fit_transform(df_movies_credits['name_belongs'])

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df_movies_credits.index, index=df_movies_credits['title']).drop_duplicates()


@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):
    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''
    
    idx = indices[titulo]
    sim_scores = list(enumerate(cosine_sim[idx]))

    for i in range(len(sim_scores)):
        for n in range(len(sim_scores[i])):
            if n%2 == 0 and sim_scores[i][n] == idx:
                sim_scores[i] = (idx,-0.01)
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:5]
    movie_indices = [i[0] for i in sim_scores]

    recomendacion =  list(df_movies_credits['title'].iloc[movie_indices])
    
    return {'lista recomendada': recomendacion}
