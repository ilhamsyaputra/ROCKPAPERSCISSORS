import random
import dash
import requests
import numpy as np
import tensorflow as tf
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from PIL import Image
from io import BytesIO
from keras.preprocessing import image
from dash.dependencies import Input, Output, State


app = dash.Dash('RH Rock Paper Scissors',
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[
                    {"name": "viewport",
                     'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}
                ])
server = app.server
app.title = 'RH Rock Paper Scissors'

rh_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

rh_model.compile(loss='categorical_crossentropy',
                 optimizer='Adam',
                 metrics=['accuracy']
                 )

rh_model.load_weights('./assets/rh_model.h5')


def prediksi(url):
    response = requests.get(url)
    image_bytes = BytesIO(response.content)

    img2 = Image.open(image_bytes)
    img2_resize = img2.resize((150, 150))
    x = image.img_to_array(img2_resize)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = rh_model.predict(images)

    if classes[0][0] == 1:
        predictions = 'Predicted as Paper'
    elif classes[0][1] == 1:
        predictions = 'Predicted as Rock'
    elif classes[0][2] == 1:
        predictions = 'Predicted as Scissors'

    return predictions, img2

def generateRandomImageUrl():
    random_image = open('./assets/url.txt').read().split("\n")[:-1]
    i = random.randint(0,len(random_image)-1)
    url = random_image[i]
    return url


## Layout
app.layout = html.Div([
    html.H1('RH Rock Paper Scissors'),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Label('Input image URL here:'),
            dbc.Input(id='input_url', type='text', value=generateRandomImageUrl()),
            html.Br(),
            dbc.Button('Run', id='proses', n_clicks=0, outline=True, color='primary',
                       style={'float': 'right', 'width': '40%'}),
            dbc.Button('Random', id='random_url', n_clicks=0, outline=True, color='primary',
                       style={'float': 'right', 'width': '40%'}),
            html.Br(),
            html.Br(),
        ]),
        dbc.Col([
            html.Center(
                html.Div(id='gambar2'),
            ),
            html.Div(id='output'),
        ])
    ]),
    html.Hr(),
    dcc.Markdown('''
    Code by [M. Ilham Syaputra](https://www.linkedin.com/in/m-ilham-syaputra/)
    '''),
    ], style={'margin-top': '1%', 'margin-bottom': '1%', 'margin-left': '10%', 'margin-right': '10%'}
)

@app.callback(
    Output('input_url', 'value'),
    Input('random_url', 'n_clicks')
)
def randomize(random_url):
    link = generateRandomImageUrl()
    return link

@app.callback(
    Output('gambar2', 'children'),
    Input('proses', 'n_clicks'),
    State('input_url', 'value'),
)
def process(proses, input_url):
    if input_url is not None:
        try:
            prediction, gambar = prediksi(input_url)

            output = html.Div([
                html.Label(input_url),
                html.Br(),
                html.Img(src=gambar, height=250, style={'box-shadow': 'rgba(0, 0, 0, 0.12) 0px 1px 3px, rgba(0, 0, 0, 0.24) 0px 1px 2px'}),
                html.H5(prediction)
            ])

            return output
        except:
            return dbc.Alert([html.H5('Something is wrong. Please use other image')], color="danger")
    else:
        return dbc.Alert([html.H5("You can't leave the url blank")], color="danger")


if __name__ == '__main__':
    app.run_server(debug=True)