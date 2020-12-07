# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from pointpats import centrography
from pyproj import Proj
from sklearn import preprocessing
import warnings
import dash_table
from dash.dependencies import Input, Output

warnings.filterwarnings('ignore')
pp = Proj(proj='utm',zone=10,ellps='WGS84', preserve_units=False)

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv("./Data_Proyecto.csv", encoding = "ISO-8859-1")

original = df


colors = {
    'red': '#E8244C',
    'lightGrey': '#ddd8d1',
    'darkGrey': '#666666',
    'brown': '#90452e'
}

#CompleteDF = pd.DataFrame(columns = ["week","suburb","numClients","totalSales","mean","UpOutliers","LowOutliers","var","iqr","dispersionKM"])

def prepareData(customer_ids, data):
  length = customer_ids.shape[0]
  data_customers = data[data['Customer'].isin(customer_ids)]
  data_customers = data_customers[['lat2', 'long2', 'Customer', 'Sales2']]
  X = data_customers.drop_duplicates()
  X.insert(0, 'id', range(0,length))
  X = X.set_index('id')
  lat_long = X[['lat2', 'long2']]
  return X, lat_long

def plotElbow(data, limit):
  sse1 = []

  for i in range(1, limit):
      kmeans1 = KMeans(n_clusters = i, random_state = 0)
      kmeans1.fit(data)
      sse1.append(kmeans1.inertia_)

  plt.plot(range(1, limit), sse1)
  plt.title('The elbow method Model 1')
  plt.xlabel('Number of clusters')
  plt.ylabel('SSE 1')
  #plt.show()

def plotSilohuette(data, limit):
  silohuette1 = np.zeros(limit - 1)
  for i in range(2, limit + 1):
      kmeans1 = KMeans(n_clusters=i, random_state = 0).fit(data)
      results1 = kmeans1.predict(data, sample_weight=None)
      silohuette1[i-2] = metrics.silhouette_score(data, results1)
      
  #plot the silhoute
  plt.plot(range(2, limit + 1), silohuette1)
  plt.title('Silohouette Coefficient 1')
  plt.xlabel('Number of clusters')
  plt.ylabel('Silohouette coefficient')
  #plt.show()

  silohuette1[0] = 0
  numClusters = np.argmax(silohuette1)
  print("best number of clusters:" , numClusters)
  
  return numClusters

def plotClusters(customer_ids, data, limit, elbow=True, silohutte=True):
  X, lat_long = prepareData(customer_ids, data)
  
  if elbow:
    plotElbow(lat_long, limit)
  
  if silohutte:
    numClusters = plotSilohuette(lat_long, limit)
  else:
    numClusters = input("number of clusters?")
    numClusters = int(numClusters)

  data_customers_predicted = []
  kmeans = KMeans(n_clusters = numClusters, random_state = 0)
  predicted = pd.Series(kmeans.fit_predict(lat_long), name='cluster')
  data_customers_predicted = pd.concat([X, predicted], axis=1)
  fig = px.scatter(data_customers_predicted, x='long2', y='lat2', color='cluster')
  #fig.show()
  return data_customers_predicted

  # Method to analyze each week and append the data to the complete DF
def AnalyzeWeek(dfWeek, apd, week):
    for i in range (0,NumClusters):
        df_col = dfWeek[dfWeek['suburb'] == i]
        xx, yy = pp(df_col["long2"].values, df_col["lat2"].values)
        df_col["X"] = xx/1000
        df_col["Y"] = yy/1000
        mean_center = centrography.mean_center(df_col[['lat2', 'long2']])
        sales = df_col.Sales2.sum()
        customerN = df_col.Sales2.count()
        mean = sales/customerN
        q75, q25 = np.percentile(df_col.Sales2, [75 ,25])
        iqr = q75 - q25
        lower_bound = q25 -(1.5 * iqr) 
        upper_bound = q75 +(1.5 * iqr)
        UpOutliers = df_col[df_col["Sales2"] > upper_bound].count()["Sales2"]
        LowOutliers = df_col[df_col["Sales2"] < lower_bound].count()["Sales2"]
        stdv = df_col.loc[:,"Sales2"].std()
        var = math.sqrt(stdv)
        disp = centrography.std_distance(df_col[['X','Y']])
        df1 = pd.DataFrame({
                        "week":[week],
                        "suburb":[i],
                        "numClients":[customerN],
                        "totalSales":[sales],
                        "mean":[mean],
                        "UpOutliers":[UpOutliers],
                        "LowOutliers":[LowOutliers],
                        "var":[var],
                        "iqr":[iqr],
                        "dispersionKM":[disp]})
        apd.append(df1)

def AnalyzeProduct(dfProd, apd):
    #For each week
    for i in range(dfProd.min()["week"],dfProd.max()["week"]+1):
        #Get just the data of that week
        dfWeek = dfProd[dfProd['week'] == i]
        #Cluster the week
        X = dfWeek[["lat2","long2"]]
        kmeans = KMeans(n_clusters = NumClusters, random_state = 0)
        dfWeek["suburb"] = kmeans.fit_predict(X)
        fig = px.scatter(dfWeek, x='long2', y='lat2', color='suburb', size='Sales2')
        #fig.show()
        AnalyzeWeek(dfWeek, apd, i)

def AnalyzeDF(df, apd):
    products = set(df['det_segment'])
    products = list(products)
    #For each product
    for prod in products:
        #Get just the data of that product
        dfProd = df[df['det_segment'] == prod]
        AnalyzeProduct(dfProd, apd)

def clustering(df):
    kmeans = KMeans(n_clusters = 86, random_state = 0)
    predicted = kmeans.fit_predict(df[['lat2','long2']])
    df['cluster'] = predicted
    df = df.reset_index()
    return df

#data = df.groupby('Customer').count()
#complete_customer_list_all = data.index

figProductsDist = px.pie(df, values='Sales2', names='ï»¿det_segment', title='Ventas de productos')

df = df.groupby('Customer').agg(
    {
        'Sales2': "sum",
        'lat2': "min",
        'long2': "min",
    }
)

original = clustering(original)

df = clustering(df).sort_values(by=['Customer'])

df_suburbs = df.groupby('cluster').agg(
    {
        'Sales2': "mean",
        'lat2': "mean",
        'long2': "mean",
    }
).sort_values(by=['cluster'])

df_suburbs = df_suburbs.reset_index()


fig = px.histogram(df, x="cluster")

fig_map = px.density_mapbox(df_suburbs, lat='lat2', lon='long2', z='Sales2', radius=10,
                        zoom=8,
                        mapbox_style="stamen-terrain")
fig_scatter = px.scatter(df_suburbs, x ="lat2", y="long2")                   

figSalesPerCapita = px.scatter(df_suburbs, x="cluster", y="Sales2", color="cluster")

figSalesDist = px.box(df_suburbs, y="Sales2", color_discrete_sequence=["red"])


productos = ['AGUA MINERAL','AGUA PURIFICADA','AGUA SABORIZADA','BEBIDAS DE FRUTA','BEBIDAS DE SOYA',
             'BEBIDAS ENERGETICAS','COLAS LIGHT','COLAS REGULAR','ISOTÃ“NICOS REGULAR','JUGOS Y NÃ‰CTARES',
             'LECHE UHT ESPECIALIZADA','LECHE UHT REGULAR','LECHE UHT SABORIZADA','SABORES LIGHT','SABORES REGULAR',
             'TÃ‰ REGULAR']


frames = []
for producto in productos:
    csv = producto+'canibalizantes.csv'    
    canibalizantes_df = pd.read_csv(csv)
    #canibalizantes_df['Customer'] = producto
    canibalizantes_df = canibalizantes_df.rename(columns={'customer': 'Customer'})
    frames.append(canibalizantes_df)

result = pd.concat(frames)
df = pd.merge(df, result, on='Customer', how='outer')

# create a list of our conditions
conditions = [
    (df['canibalizado'] == True) & (df['canibalizador'] == False),
    (df['canibalizador'] == True) & (df['canibalizado'] == False),
    (df['canibalizador'] == True) & (df['canibalizado'] == True),
    (df['canibalizador'].isnull()) & (df['canibalizado'].isnull()),
    ]

# create a list of the values we want to assign for each condition
values = ['canibalizado','canibalizador','ambos','ninguno']

# create a new column and use np.select to assign values to it using our lists as arguments
df['clasificacion'] = np.select(conditions, values)


figUbicaciones = px.scatter(df, x="long2", y="lat2", color="clasificacion")

grouped_multiple = original[original['cluster'] == 21]
grouped_multiple = grouped_multiple.groupby(['Customer', 'week']).agg({
    'Sales2': "mean",
    'lat2': "mean",
    'long2': "mean",
    })
grouped_multiple = grouped_multiple.reset_index()
#print(grouped_multiple.head())

figSalesByWeek = px.line(grouped_multiple, x="week", y="Sales2", color='Customer')

app.layout = html.Div(children=[
    html.Div(children=[
        html.H1(children='Canibalización entre clientes',
        style={
            'color': colors['red'],
            'text-align': 'center'
        },
        className="col-12"
        ),

        html.Div(children='''
            Efectos de canibalización entre clientes de Arca Continental a lo largo de 11 semanas del año 2020.
        ''',
        style={
            'color': colors['darkGrey'],
            'text-align': 'center'
        },
        className="col-12"
        ),
    ],
    className="row p-5"
    ),

    html.Div(children=[
        html.H1(children="¿Cuánto es el total de ventas per capita en las diferentes colonias?")
    ],
    style={
        'background-color': colors['red'],
        'color': 'white',
        'height': '300px',
        'text-align': 'center',
        'display': 'flex',
        'justify-content': 'center',
        'align-content': 'center',
    },
    className="row p-5"
    ),

    html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='example-graph',
                figure=figSalesDist 
            ),
        ],
        className="col-lg-6"
        ),

        html.Div(children=[
            html.Div(children=[
                html.H6("Promedio de ventas totales per capita en las diferentes colonias durante 11 semanas")
            ]
            ),

            dash_table.DataTable(
                id='table',
                columns=[{"name": "Colonia", "id": "cluster"}, {"name": "Ventas","id": "Sales2"}],
                data=df_suburbs.to_dict('records'),
                page_size=10,
            )
        ],
        className="col-lg-6"
        ),

    ],
    className="row p-5 my-5"
    ),

    html.Div(children=[
        html.H1(children="20 productos")
    ],
    style={
        'background-color': colors['red'],
        'color': 'white',
        'height': '300px',
        'text-align': 'center',
        'display': 'flex',
        'justify-content': 'center',
        'align-content': 'center',
    },
    className="row p-5"
    ),

    html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='products',
                figure=figProductsDist
            ),
            
        ],
        className="col-12"
        ),

    ],
    className="row p-5 my-5"
    ),



    html.Div(children=[
        html.H1(children="11,365 clientes")
    ],
    style={
        'background-color': colors['red'],
        'color': 'white',
        'height': '300px',
        'text-align': 'center',
        'display': 'flex',
        'justify-content': 'center',
        'align-content': 'center',
    },
    className="row p-5"
    ),

    html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='ubicaciones',
                figure=figUbicaciones
            ),
            
        ],
        className="col-12"
        ),

    ],
    className="row p-5 my-5"
    ),

     html.Div(children=[
        html.H1(children="Ventas por semana")
    ],
    style={
        'background-color': colors['red'],
        'color': 'white',
        'height': '300px',
        'text-align': 'center',
        'display': 'flex',
        'justify-content': 'center',
        'align-content': 'center',
    },
    className="row p-5"
    ),
    
    html.H6(children="cluster 21"),

    html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='fig-sales-week1',
                figure=figSalesByWeek
            ),

            
            
        ],
        className="col-12"
        ),

    ],
    className="row p-5 my-5"
    ),


])

if __name__ == '__main__':
    app.run_server(debug=True)