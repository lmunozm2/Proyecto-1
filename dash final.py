import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import numpy as np
from scipy.stats import t
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

data = pd.read_excel("data nuevo1.xlsx", index_col=0)
var_continuas=["targeted_productivity","smv","wip","over_time","incentive","idle_time","idle_men","no_of_style_change", "no_of_workers","actual_productivity"]
var_categoricas=["day", "quarter","department","team"]
df_continuas=data[var_continuas]
df_categoricas=data[var_categoricas]
df_categoricas_numericas = pd.get_dummies(df_categoricas, dtype="float64")
data_final = pd.concat([df_continuas, df_categoricas_numericas], axis=1)

features =  ["smv", "incentive", "idle_men", "no_of_style_change", "department_finishing"]
X=data_final[features]
y = data_final["actual_productivity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Configurar la aplicación Dash
app = dash.Dash(__name__)

#INTERVALO DE CONFIANZA
def confidence_interval(prediction, input_data, X_train, y_train, confidence=0.95):
    # Obtener las características relevantes del DataFrame de entrada
    features = [feature for feature in X_train.columns if feature in input_data.columns]
    # Filtrar los datos de entrenamiento para que solo contengan las características relevantes
    X_train_filtered = X_train[features]
    
    # Calcular la predicción correspondiente a los datos de entrada proporcionados por el usuario
    y_pred = linreg.predict(input_data)
    
    # Calcular la varianza de los errores residuales usando los datos de entrenamiento filtrados
    y_pred_train = linreg.predict(X_train_filtered)
    residual_errors = y_train - y_pred_train
    s_residual = np.sqrt(np.sum(residual_errors ** 2) / (len(y_train) - 2))
    
    # Calcular el valor t de Student para el nivel de confianza deseado
    alpha = 1 - confidence
    t_value = t.ppf(1 - alpha / 2, len(y_train) - 2)
    
    # Calcular el intervalo de confianza
    confidence_interval_lower = y_pred - t_value * s_residual
    confidence_interval_upper = y_pred + t_value * s_residual
    
    return confidence_interval_lower, confidence_interval_upper

#PROMEDIOS
data_numeric = data.select_dtypes(include=['int', 'float'])
promedios = data_numeric.mean().round(3)
nombres_features =  ["smv", "incentive", "idle_men", "no_of_style_change"]
nombres_personalizados = ["Minutos Estándar", "Incentivo", "Trabajadores Inactivos", "Cambios de Estilo"]
unidades = ["Minutos", "Unidades monetarias", "Trabajadores", "Cambios"]
tabla_promedios = html.Table([
    html.Thead(
        html.Tr([html.Th("Variable"), html.Th("Valor Promedio"), html.Th("Unidades")])
    ),
    html.Tbody([
        html.Tr([
            html.Td(nombres_personalizados[i], style={'text-align': 'center'}),
            html.Td(f'{promedios[nombres_features[i]]:.3f}', style={'text-align': 'center'}),
            html.Td(unidades[i], style={'text-align': 'center'})
        ]) for i in range(len(nombres_features))
    ])
],
style={'margin':'auto'}
)

#BARRAS
# Calcular la productividad promedio por departamento
productivity_by_department = data.groupby('department')['actual_productivity'].mean()
# Crear el gráfico de barras
fig_productivity_by_department = go.Figure()
fig_productivity_by_department.add_trace(go.Bar(
    x=productivity_by_department.index,
    y=productivity_by_department.values,
    text=productivity_by_department.values.round(2), 
    textposition='auto',
    marker=dict(color=['blue', 'red']),
))
# Actualizar diseño del gráfico
fig_productivity_by_department.update_layout(
    title='Productividad Promedio por Departamento',
    xaxis_title='Departamento',
    yaxis_title='Productividad Promedio'
)

# Diseño del dashboard
app.layout = html.Div([
    html.H1("Desempeño de la Empresa", style={'text-align': 'center','color': 'blue', 'font-family': 'Verdana'}),
    
    html.Div([
        #SMV
        html.Div([
            html.Label("Valor del minuto estándar:", style={'font-wight':'bold'}),
            dcc.Input(
                id='input-smv',
                type='number',
                value=round(min(X['smv']) + max(X['smv']) / 2,1),
                step=0.1,
                min=min(X['smv']),
                max=max(X['smv']),
                style={'width': '200px', 'margin-bottom': '10px','font-family': 'Verdana'}
            )
        ]),
        #INCENTIVO
        html.Div([
            html.Label("Incentivo financiero:", style={'font-wight':'bold'}), 
            dcc.Input(
                id='input-incentive',
                type='number',
                value=(min(X['incentive']) + max(X['incentive'])) / 2,
                step=0.1,
                min=min(X['incentive']),
                #max=max(X['incentive']),
                style={'width': '200px', 'margin-bottom': '10px', 'font-family': 'Verdana'}
            )
        ]),
        #idle men
        html.Div([
            html.Label("Número de trabajadores inactivos por la interrupción de la producción:", style={'font-wight':'bold'}),  
            dcc.Input(
                id='input-idle_men',
                type='number',
                value=round((min(X['idle_men']) + max(X['idle_men'])) / 2,0),
                step=1,
                min=min(X['idle_men']),
                max=max(X['idle_men']),
                style={'width': '200px', 'margin-bottom': '10px','font-family': 'Verdana'}
            )
        ]),
        #CAMBIOS
        html.Div([
            html.Label("Número de cambios en el estilo de un producto:", style={'font-wight':'bold'}), 
            dcc.Input(
                id='input-no_of_style_change',
                type='number',
                value=round((min(X['no_of_style_change']) + max(X['no_of_style_change'])) / 2,0),
                step=1,
                min=0,
                max=8,
                style={'width': '200px', 'margin-bottom': '10px','font-family': 'Verdana'}
            )
        ]),
    ]),
    # DEPARTMENTO FINISHING LISTA DESPLEGABLE
    html.Div([
        html.Label("Departamento:", style={'font-weight':'bold'}),
        dcc.Dropdown(
            id='dropdown-department_finishing',
            options=[
                {'label': 'Finishing', 'value': 1},  
                {'label': 'Sweing', 'value': 0} 
            ],  
            value=1,  # Valor predeterminado
            style={'width': '200px','font-family': 'Verdana'} 
        )
    ], style={'margin-bottom': '20px'}),
    
    #TITULOS Y DESPLIEGUE DE INFORMACIÓN
    html.H2("Predicción de Productividad", style={'text-align': 'center', 'color': 'green','font-family': 'Verdana'}),
    html.Div(id='productivity-output', style={'font-size': '20px', 'text-align': 'center','font-family': 'Verdana'}),
    html.H3("Valores promedio", style={'text-align': 'center','color': 'green','font-family': 'Verdana'}),
    tabla_promedios,
    html.H3("Graficos de interes",style={'text-align': 'center', 'color': 'green','font-family': 'Verdana'}),
    # Gráfico de pastel para la distribución por departamento
    dcc.Graph(id='productivity-by-department',figure=fig_productivity_by_department),
    dcc.Graph(id = 'graph-prediction'),
    html.Div(id='prediction-interval-output', style={'text-align': 'center', 'font-family': 'Verdana'})    
    ], style={'margin': '20px', 'padding': '20px', 'border': '1px solid black','font-family': 'Verdana'})



# Callback para actualizar la predicción de productividad cuando se cambian los controles
@app.callback(
    Output('productivity-output', 'children'),
    Output('graph-prediction','figure'),
    [Input('input-' + feature, 'value') for feature in features if feature != "department_finishing"] +
    [Input('dropdown-department_finishing', 'value')]
)
def update_productivity_output(*args):
     # Obtener los valores seleccionados en los controles
    input_values = args[:-1]
    department_finishing_value = args[-1]
    
    # Preparar los datos para la predicción
    input_data = dict(zip([feature for feature in features if feature != "department_finishing"], input_values))
    input_data["department_finishing"] = department_finishing_value
    df = pd.DataFrame(input_data, index=[0])
    
    # Realizar la predicción con el modelo
    prediction = linreg.predict(df)
    
    #Calcular el intervalo de confianza 
    confidence_interval_lower, confidence_interval_upper = confidence_interval(prediction, df, X_train, y_train)
    
    #Que muestre la prediccion y el intervalo de confianza 
    output_prediccion = f'La productividad sería de: {round(prediction[0],4)*100}%'
    output_intervalo = f'Intervalo de confianza: [{round(confidence_interval_lower[0] * 100, 2)}, {round(confidence_interval_upper[0] * 100, 2)}]'
        
        
    # Crear el gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[confidence_interval_lower[0], confidence_interval_upper[0]], y=['Prediction', 'Prediction'],
                             fill='tonextx', fillcolor='rgba(173, 216, 230, 0.5)', line=dict(color='rgba(173, 216, 230, 1)', width=6), name='Intervalo de confianza'))
    fig.add_trace(go.Scatter(x=[prediction[0] - 0.05, prediction[0] + 0.05], y=['Prediction', 'Prediction'],
                             mode='lines', line=dict(color='rgba(0, 128, 0, 1)', width=8), name='Predicción'))
    fig.update_layout(title='Intervalo de confianza con la predicción',
                      xaxis_title='Productividad',
                      yaxis_title='Predicción',
                      showlegend=True,
                      legend=dict(x=0, y=1))
    
    return [html.Div(output_prediccion), html.Div(output_intervalo)], fig
if __name__ == '__main__':
    app.run_server(debug=True)