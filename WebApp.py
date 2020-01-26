#Data
import pandas as pd
import numpy as np
import pickle


#Dash visualización
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.express as px
import json



df = pd.read_json("/home/alex/Desktop/Mushroom challenges/df_json_file.json")


body = dbc.Container(
    [
        dbc.Row([   
            dbc.Col([html.H1("Mushroom Challenge"),
                    html.P("Please enter your parameters."),
                    html.Label(["Name of the mushroom",
                    dcc.Input(
                                id = "Name_Input",
                                placeholder="Enter a name...",  
                                type="text",
                                style={
                                        "width": "230px",
                                        "margin-left" : "30px"
                               }),]),
                    
                ]),
            dbc.Col([html.Img(src="https://idp.evonik.com/nidp/evonik/misc/img/logo.png",
                            style={
                        'height': '75%',
                        'width': '75%',
                        'float': 'right',
                        'position': 'relative',
                        'padding-top': 10,
                        'padding-left':30,
                        'padding-right': 0
                                },
                            )]),
                ]),
        dbc.Row([
                dbc.Col([
                        html.Label(["Odor",
                        dcc.Dropdown(id="odor_dropdown",
                            options=[
                        {'label': 'Almond', 'value': 'a'},
                        {'label': 'Anise', 'value': 'l'},
                        {'label': 'creosote', 'value': 'c'},
                        {'label': 'fishy', 'value': 'y'},
                        {'label': 'foul', 'value': 'f'},
                        {'label': 'musty', 'value': 'm'},
                        {'label': 'none', 'value': 'n'},
                        {'label': 'pungent', 'value': 'p'},
                        {'label': 'spicy', 'value': 's'},
                    ],
                    style={"margin-right":"35px",
                    "verticalAlign":"middle"},
                    placeholder="Select an Odor",
                        )]),
                        html.Label(["Cap_Surface",
                        dcc.Dropdown(id="cap_surface_dropdown",
                            options=[ 
                        {'label': 'fibrous', 'value': 'f'},
                        {'label': 'grooves', 'value': 'g'},
                        {'label': 'smooth', 'value': 's'},
                        {'label': 'scaly', 'value': 'y'},                     
                    ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Cap-Surface",
                        )]),
                        html.Label(["Gill-Attachment", 
                        dcc.Dropdown(id="gill_attachment_dropdown",
                            options=[
                        {'label': 'attached', 'value': 'a'},
                        {'label': 'free', 'value': 'f'},
                    ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Gill-Attachment",
                        )]),
                        html.Label(["Population", 
                        dcc.Dropdown(id="population_dropdown",
                            options=[
                        {'label': 'abundant', 'value': 'a'},
                        {'label': 'clustered', 'value': 'c'},
                        {'label': 'numerous', 'value': 'n'},
                        {'label': 'scattered', 'value': 's'},
                        {'label': 'several', 'value': 'v'},
                        {'label': 'solitary', 'value': 'y'},                     
                    ],
                    style={"margin-right":"35px"},
                    placeholder="Select a population",
                        )]),
                        html.Label(["Stalk Surface Above Ring", 
                        dcc.Dropdown(id="stalk_surface_above_ring_dropdown",
                            options=[
                        {'label': 'fibrous', 'value': 'f'},
                        {'label': 'silky', 'value': 'k'},
                        {'label': 'smooth', 'value': 's'},
                        {'label': 'scaly', 'value': 'y'},
                    ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Stalk Surface Above Ring",
                        )]),
                        html.Label(["Gill-Spacing", 
                        dcc.Dropdown(id="gill_spacing_dropdown",
                            options=[
                        {'label': 'close', 'value': 'c'},
                        {'label': 'crowded', 'value': 'w'},
                    ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Gill-Spacing",
                        )]),
                        html.Label(["Gill-Size", 
                        dcc.Dropdown(id="gill_size_dropdown",
                            options=[
                        {'label': 'broad', 'value': 'b'},
                        {'label': 'narrow', 'value': 'n'},
                    ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Gill-Size",
                        )]),
                dbc.Row([
                        html.Label(["Stalk-Root", 
                        dcc.Dropdown(id="stalk_root_dropdown",
                            options=[
                        {'label': 'bulbous', 'value': 'b'},
                        {'label': 'club', 'value': 'c'},
                        {'label': 'equal', 'value': 'e'},
                        {'label': 'rooted', 'value': 'r'},
                        {'label': 'missing', 'value': '?'},
                        ],
                    style={"margin-right":"35px",
                    "verticalAlign":"middle"},
                    placeholder="Select a Stalk-Root",
                        )]),
                        html.Label(["Ring-Type", 
                        dcc.Dropdown(id="ring_type_dropdown",
                            options=[
                        {'label': 'evanescent', 'value': 'e'},
                        {'label': 'flaring', 'value': 'f'},
                        {'label': 'large', 'value': 'l'},
                        {'label': 'none', 'value': 'n'},
                        {'label': 'pendant', 'value': 'p'},
                        ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Ring-Type",
                        )]),
                        html.Label(["Veil-Color", 
                        dcc.Dropdown(id="veil_color_dropdown",
                            options=[
                        {'label': 'yellow', 'value': 'y'},
                        {'label': 'white', 'value': 'w'},
                        {'label': 'orange', 'value': 'o'},
                        {'label': 'brown', 'value': 'n'},
                    ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Veil-Color",
                        )]),
                        html.Label(["Stalk-Color-Above-Ring", 
                        dcc.Dropdown(id="stalk_color_above_ring_dropdown",
                            options=[
                        {'label': 'brown', 'value': 'n'},
                        {'label': 'buff', 'value': 'b'},
                        {'label': 'cinnamon', 'value': 'c'},
                        {'label': 'gray', 'value': 'g'},
                        {'label': 'orange', 'value': 'o'},
                        {'label': 'pink', 'value': 'p'},
                        {'label': 'red', 'value': 'e'},
                        {'label': 'white', 'value': 'w'},
                        {'label': 'yellow', 'value': 'y'},
                        ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Stalk-Color-Above-Ring",
                        )]),
                        html.Label(["Spore-Print-Color", 
                        dcc.Dropdown(id="spore_print_color_dropdown",
                            options=[
                        {'label': 'black', 'value': 'k'},
                        {'label': 'brown', 'value': 'n'},
                        {'label': 'buff', 'value': 'b'},
                        {'label': 'chocolate', 'value': 'h'},
                        {'label': 'green', 'value': 'r'},
                        {'label': 'orange', 'value': 'o'},
                        {'label': 'purple', 'value': 'u'},
                        {'label': 'white', 'value': 'w'},
                        {'label': 'yellow', 'value': 'y'},
                        ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Spore-Print-Color",
                        )]),
                        html.Label(["ring-number", 
                        dcc.Dropdown(id="ring_number_dropdown",
                            options=[
                        {'label': 'none', 'value': 'n'},
                        {'label': 'one', 'value': 'o'},
                        {'label': 'two', 'value': 't'},
                        ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Ring-Number",
                        )]),
                        dbc.Button("Submit",
                        color="secondary",
                        id='submit-button',
                        n_clicks=0,
                        style={
                            "width": "100px",
                            "margin-left": "30px",
                            "margin-right":"10px"
                               }),
                            ]),
                    ]),
    ]),
            dbc.Row([
                html.Div(
                        id = "prediction",
                style={
                    "margin-top":"30px",
                    "margin-bottom":"30px",
                    "width": "800px",
                    "margin-top" : "55px",
                    "margin-left": "30px"
                               }
                        )]),
            dbc.Row([
                    # Hidden div inside the app that stores the intermediate value
                    html.Div(id='intermediate-value', style={'display': 'none'}),
                    dash_table.DataTable(
                                            id='table',
                                            columns=[{"name": i, "id": i} for i in df.columns],
                                            data=df.to_dict('records'),
                                            style_table={'overflowX': 'scroll',
                                                         'overflowY':'scroll'},
                                            export_format='csv',
                                            export_headers='display',
                                            merge_duplicate_headers=True
                                        ),
                    #dbc.Table.from_dataframe(id='table',df)
                    ]),     
])


#Launching the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([body])


@app.callback(
    [Output('prediction', 'children'),
    Output('intermediate-value',"children")],
    [Input("submit-button","n_clicks")],
    state=[
    State('Name_Input',"value"),
    State('odor_dropdown', 'value'),
    State('cap_surface_dropdown', 'value'),
    State('gill_attachment_dropdown', 'value'),
    State('population_dropdown', 'value'),
    State('stalk_surface_above_ring_dropdown', 'value'),
    State('gill_spacing_dropdown', 'value'),
    State('gill_size_dropdown', 'value'),
    State('stalk_root_dropdown', 'value'),
    State('ring_type_dropdown', 'value'),
    State('veil_color_dropdown', 'value'),
    State('stalk_color_above_ring_dropdown', 'value'),
    State('spore_print_color_dropdown', 'value'),
    State('ring_number_dropdown','value')]
            )
def update_output(n_clicks,
                    Name_Input,
                    odor_dropdown,
                    cap_surface_dropdown,
                    gill_attachment_dropdown,
                    population_dropdown,
                    stalk_surface_above_ring_dropdown,
                    gill_spacing_dropdown,
                    gill_size_dropdown,
                    stalk_root_dropdown,
                    ring_type_dropdown,
                    veil_color_dropdown,
                    stalk_color_above_ring_dropdown,
                    spore_print_color_dropdown,
                    ring_number_dropdown):

    #Creating the dataframe...

    if n_clicks is None:
        raise PreventUpdate
    else:
        data = {
                "name"                      :Name_Input,
                "prediction"                :"",
                'veil-color'                :veil_color_dropdown,
                'stalk-surface-above-ring'  :stalk_surface_above_ring_dropdown,
                'stalk-root'                :stalk_root_dropdown,
                'stalk-color-above-ring'    :stalk_color_above_ring_dropdown,
                'spore-print-color'         :spore_print_color_dropdown,
                'ring-type'                 :ring_type_dropdown,
                'ring-number'               :ring_number_dropdown,
                'population'                :population_dropdown,
                "odor"                      :odor_dropdown,
                'gill-spacing'              :gill_spacing_dropdown,
                'gill-size'                 :gill_size_dropdown,
                'gill-attachment'           :gill_attachment_dropdown,
                "cap-surface"               :cap_surface_dropdown,
                }

    df_model = pd.DataFrame([data])

    #Loading the pipeline and the model
    loading_model = pickle.load(open("/home/alex/Desktop/Mushroom challenges/finalized_model.sav", 'rb'))

    predicted_value = loading_model.predict(df_model.drop(columns=["name","prediction"]))

    predicted_value = ["Poisonous" if x == 1 else "edible" for x in predicted_value][0]

    df_model.loc[:,"prediction"] = predicted_value


    if predicted_value == "edible":
        alert = [dbc.Alert([html.H1(f"The mushroom {Name_Input} is edible!",className="alert-heading")],color="success")]
        return alert, df_model.to_json(orient='split')
    else:
        alert = [dbc.Alert([html.H1(f"The mushroom {Name_Input} is Poisonous!",className="alert-heading")],color="danger")]
        return alert,df_model.to_json(orient='split')

@app.callback(
            [Output('table', 'data'),
            Output('table', 'columns')],
            [Input('intermediate-value', 'children')])
def update_table(jsonified_cleaned_data):
    df_new_row = pd.read_json(jsonified_cleaned_data, orient='split')
    df = pd.read_json("/home/alex/Desktop/Mushroom challenges/df_json_file.json")
    df = df.append(df_new_row,ignore_index=True, verify_integrity=False, sort=None)
    df.to_json("/home/alex/Desktop/Mushroom challenges/df_json_file.json")
    return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]


if __name__ == '__main__':
    app.run_server(debug=True)