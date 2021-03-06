#Data
import pandas as pd
import numpy as np
import pickle
import json

#Dash visualización
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash.dash import no_update

#Sckit-Learn packages
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

#Plotly charts
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.express as px

#Getting all the previous import from before
from app import app


###############################################################################
########### LANDING PAGE LAYOUT ###########
###############################################################################


df = pd.read_json("output/df_json_file.json")
#Mapping the variables with the full names
def mapping_df(df):

    odor = {'a': 'Almond',
            'l': 'Anise',
            'c': 'creosote',
            'y': 'fishy',
            'f': 'foul',
            'm': 'musty',
            'n': 'none',
            'p': 'pungent',
            's': 'spicy'}

    cap_surface = { 'f': 'fibrous',
                    'g': 'grooves',
                    's': 'smooth',
                    'y': 'scaly'}
    gill_attachment = { 'a': 'attached',
                        'f': 'free'}

    population = {  'a': 'abundant',
                    'c': 'clustered',
                    'n': 'numerous',
                    's': 'scattered',
                    'v': 'several',
                    'y': 'solitary'}

    stalk_surface_above_ring = { 'f': 'fibrous',
                                'k': 'silky',
                                's': 'smooth',
                                'y': 'scaly'}

    gill_spacing = {'c': 'close',
                    'w': 'crowded'}

    gill_size = { 'b': 'broad',
                'n': 'narrow'}           

    stalk_root = { 'b': 'bulbous',
                'c': 'club',
                    'e': 'equal',
                    'r': 'rooted',
                    '?': 'missing'}

    ring_type = {'e': 'evanescent',
                'f': 'flaring',
                'l': 'large',
                'n': 'none',
                'p': 'pendant'}    

    veil_color = {'y': 'yellow',
                'w': 'white',
                'o': 'orange',
                'n': 'brown'}

    stalk_color_above_ring = {  'n': 'brown',
                                'b': 'buff',
                                'c': 'cinnamon',
                                'g': 'gray',
                                'o': 'orange',
                                'p': 'pink',
                                'e': 'red',
                                'w': 'white',
                                'y': 'yellow'}
    spore_print_color= {'k': 'black',
                        'n': 'brown',
                        'b': 'buff',
                        'h': 'chocolate',
                        'r': 'green',
                        'o': 'orange',
                        'u': 'purple',
                        'w': 'white',
                        'y': 'yellow'}

    ring_number = {'n': 'none',
                'o': 'one',
                't': 'two'}
    df.columns = df.columns.str.replace("-","_")

    df["veil_color"] = df["veil_color"].map(veil_color)
    df["stalk_surface_above_ring"] = df["stalk_surface_above_ring"].map(stalk_surface_above_ring)
    df["stalk_root"] = df["stalk_root"].map(stalk_root)
    df["stalk_color_above_ring"] = df["stalk_color_above_ring"].map(stalk_color_above_ring)
    df["spore_print_color"] = df["spore_print_color"].map(spore_print_color)
    df["ring_type"] = df["ring_type"].map(ring_type)
    df["ring_number"] = df["ring_number"].map(ring_number)
    df["population"] = df["population"].map(population)
    df["odor"] = df["odor"].map(odor)
    df["gill_spacing"] = df["gill_spacing"].map(gill_spacing)
    df["gill_size"] = df["gill_size"].map(gill_size)
    df["gill_attachment"] = df["gill_attachment"].map(gill_attachment)
    df["cap_surface"] = df["cap_surface"].map(cap_surface)

    df.columns = df.columns.str.replace("_","")

    return df

df = mapping_df(df)

body = dbc.Container(
    [
        dbc.Row([   
            dbc.Col([html.H1("Mushroom Challenge",
                            style={'color': '#800080'}),
                    html.Hr(),
                    html.H4("Please enter your parameters below"),
                    html.Hr(),
                    html.Label(["Name of the mushroom",
                    dcc.Input(
                                id = "Name_Input",
                                placeholder="Enter a name...",  
                                type="text",
                                style={
                                        "width": "230px",
                                        "margin-left" : "30px"
                               }),
                    html.Hr(),
                    ]),
                    
                ]),
            dbc.Col([html.Img(src="./assets/evonik_cropped.png",
                            style={
                        'float': 'right',
                        'position': 'relative',
                        'padding-top': "10px",
                        #'padding-bottom':'50px',
                        #'padding-left':0,
                        #'padding-right': 0
                                },
                            )]),
                ]),
#Layout with all the dropdown menus
        dbc.Row([
                dbc.Col([html.Label(["Odor",
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
                        )])]),
                dbc.Col([html.Label(["Cap Surface",
                        dcc.Dropdown(id="cap_surface_dropdown",
                            options=[ 
                        {'label': 'fibrous', 'value': 'f'},
                        {'label': 'grooves', 'value': 'g'},
                        {'label': 'smooth', 'value': 's'},
                        {'label': 'scaly', 'value': 'y'},                     
                    ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Cap-Surface",
                        )])]),
                    dbc.Col([html.Label(["Gill Attachment", 
                        dcc.Dropdown(id="gill_attachment_dropdown",
                            options=[
                        {'label': 'attached', 'value': 'a'},
                        {'label': 'free', 'value': 'f'},
                    ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Gill-Attachment",
                        )])]),
                    dbc.Col([html.Label(["Population", 
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
                        )])]),
                    dbc.Col([html.Label(["StalkSurfaceAboveRing", 
                        dcc.Dropdown(id="stalk_surface_above_ring_dropdown",
                            options=[
                        {'label': 'fibrous', 'value': 'f'},
                        {'label': 'silky', 'value': 'k'},
                        {'label': 'smooth', 'value': 's'},
                        {'label': 'scaly', 'value': 'y'},
                    ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Stalk Surface Above Ring",
                        )])]),
                    dbc.Col([html.Label(["Gill Spacing", 
                        dcc.Dropdown(id="gill_spacing_dropdown",
                            options=[
                        {'label': 'close', 'value': 'c'},
                        {'label': 'crowded', 'value': 'w'},
                    ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Gill-Spacing",
                        )])]),
                    dbc.Col([html.Label(["Gill Size", 
                        dcc.Dropdown(id="gill_size_dropdown",
                            options=[
                        {'label': 'broad', 'value': 'b'},
                        {'label': 'narrow', 'value': 'n'},
                    ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Gill-Size",
                        )])])],no_gutters=True),
                html.Hr(),  
                dbc.Row([
                    dbc.Col([html.Label(["Stalk Root", 
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
                        )])]),
                    dbc.Col([html.Label(["Ring Type", 
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
                        )])]),
                    dbc.Col([html.Label(["Veil Color", 
                        dcc.Dropdown(id="veil_color_dropdown",
                            options=[
                        {'label': 'yellow', 'value': 'y'},
                        {'label': 'white', 'value': 'w'},
                        {'label': 'orange', 'value': 'o'},
                        {'label': 'brown', 'value': 'n'},
                    ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Veil-Color",
                        )])]),
                    dbc.Col([html.Label(["StalkColorAboveRing", 
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
                        )])]),
                    dbc.Col([html.Label(["SporePrintColor", 
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
                        )])]),
                    dbc.Col([html.Label(["Ring Number", 
                        dcc.Dropdown(id="ring_number_dropdown",
                            options=[
                        {'label': 'none', 'value': 'n'},
                        {'label': 'one', 'value': 'o'},
                        {'label': 'two', 'value': 't'},
                        ],
                    style={"margin-right":"35px"},
                    placeholder="Select a Ring-Number",
                        )])]),
                    dbc.Col([
                        dbc.Button("Submit",
                        color="secondary",
                        id='submit-button',
                        n_clicks=0,
                        style={
                            "width": "100px",
                            "margin-top":"22px"
                               }),
                            ]),
                        ],no_gutters=True),
            html.Hr(),
            dbc.Row([
                html.Div(
                    dbc.Alert([html.H1(f"Please input all your parameters to get the prediction!",className="alert-heading")],color="light"), #First reminder message
                        id = "prediction",
                style={
                    "margin-top":"30px",
                    "margin-bottom":"30px",
                    "width": "1200px",
                    "margin-left": "0px"
                               }
                        )]),
            dbc.Row([
                    # Hidden div inside the app that stores the intermediate value
                    dbc.Col([html.Div(id='intermediate-value', style={'display': 'none'}),
                    dash_table.DataTable(
                                            id='table',
                                            columns=[{"name": i, "id": i} for i in df.columns],
                                            data=df.to_dict('records'),
                                            filter_action='native',
                                            sort_action='native',
                                            editable=True,
                                            style_table={'overflowX': 'scroll',
                                                         #'overflowY':'scroll',
                                                         'border': 'thin lightgrey solid',
                                                         "minWidth": '100%',
                                                         "max-width": "1200px",
                                                         'maxHeight': '300px'},
                                            export_format='csv',
                                            export_headers='display',
                                            merge_duplicate_headers=True),
                            ])
                    ]),    
])


#The layout of the app
layout = html.Div([body])

#Getting all the data needed for the prediction through State
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
def update_output(  n_clicks,
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

    if n_clicks == 0: #If n_clicks == 0 then do nothing
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

        df_model = pd.DataFrame([data]) #Loading the data into a dataframe

        #Loading the pipeline and the model
        loading_model = pickle.load(open("input/finalized_model.sav", 'rb')) 
        try:
            predicted_value = loading_model.predict(df_model.drop(columns=["name","prediction"]))
        
            predicted_value = ["Poisonous" if x == 1 else "edible" for x in predicted_value][0]
            df_model.loc[:,"prediction"] = predicted_value

            if Name_Input is None or Name_Input == "": #Alert message in case of missing name
                alert = [dbc.Alert([html.H1(f"Please enter a name!",className="alert-heading")],color="danger")]
                return alert,None

            elif predicted_value == "edible":
                alert = [dbc.Alert([html.H1(f"The mushroom {Name_Input} is edible!",className="alert-heading")],color="success")]
                return alert, df_model.to_json(orient='split')
            else:
                alert = [dbc.Alert([html.H1(f"The mushroom {Name_Input} is Poisonous!",className="alert-heading")],color="danger")]
                return alert,df_model.to_json(orient='split')
        
        except ValueError: #Alert message in case of missing fields

            alert = [dbc.Alert([html.H1(f"Please fill all the fields before submitting!",className="alert-heading")],color="danger")]
            return alert, None


#Updating the table through the Json file
@app.callback(
            [Output('table', 'data')],
            [Input('intermediate-value', 'children')])

def update_table(jsonified_cleaned_data):

    if jsonified_cleaned_data is None: #If jsonified_cleaned_data is empty then do nothing
        raise PreventUpdate
    
    #Creating a new dataframe for the new data
    df_new_row = pd.read_json(jsonified_cleaned_data, orient='split')

    df = pd.read_json("output/df_json_file.json")

    #Adding the new row to the pre-existing data
    df = df.append(df_new_row,ignore_index=True, verify_integrity=False, sort=None)

    #Save the new data
    df.to_json("output/df_json_file.json")

    #Mapping the name for display
    df = mapping_df(df)

    return [df.to_dict('records')] #Return the data
