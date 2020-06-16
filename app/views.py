from app import app
from app.covid import Covid
from flask import Flask, jsonify, redirect, url_for, render_template,session, escape, request
import os
from os import listdir
from os.path import isfile, join
from werkzeug.utils import secure_filename
import folium
import pandas as pd
from folium.plugins import MarkerCluster
import geocoder
from selenium import webdriver
from time import sleep
import re
from datetime import datetime
import smtplib

UPLOAD_FOLDER = '/Mini/app/data/'
app.secret_key = '/\x8c\x9a\xadT\xdf\x1b\xf0\r\x87\xa9\x1aV\xd5\x04\xbc\x0c\xff|\x15\x0edmd'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_PATH = '/Mini/app/models/covid_model.h5'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/index.html', methods=['GET'])
def index():
    # Main page
    session.pop('filename', None)
    heatmap_path='app/static/heatmap/'
    heatmap_files=[f for f in listdir(heatmap_path) if isfile(join(heatmap_path, f))]
    [os.remove(heatmap_path+heatmap_file) for heatmap_file in heatmap_files if heatmap_file != 'heatmap.jpeg']
    data_files=[f for f in listdir(UPLOAD_FOLDER) if isfile(join(UPLOAD_FOLDER, f))]
    [os.remove(UPLOAD_FOLDER+data_file) for data_file in data_files if data_file != 'covid.jpg']
    return render_template('index.html')

@app.route('/heatmap.html', methods=['GET'])
def heatmap():
    FILE=None
    if isfile('/Mini/app/static/heatmap/'+str(session['filename'])):
        FILE=escape(session['filename'])
    return render_template('heatmap.html',image_path=FILE)

@app.route('/', methods=['GET'])
def landing():
    return render_template('landing.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            resp = jsonify({'message': 'No file part in the request'})
            resp.status_code = 400
            return resp
        file = request.files['file']
        if file.filename == '':
            resp = jsonify({'message': 'No file selected for uploading'})
            resp.status_code = 400
            return resp
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file = str(UPLOAD_FOLDER+filename)
            if isfile(file) == False:
                return jsonify({'message': 'File not found'})
            preds = Covid(file, MODEL_PATH).covid_predict()
            if preds['prediction'] =='Negative':
                session['filename']=None
            else:
                session['filename']=filename
            os.remove(file)
            return preds
        else:
            resp = jsonify(
                {'message': 'Allowed file types are jpg, jpeg, png'})
            resp.status_code = 400
            return resp
    return redirect('/')

@app.route('/map.html')
def map():
    dataset = pd.read_csv('/Mini/lab_coordinate.csv')
    place = dataset[ ['Latitude', 'Longitude'] ]
    place=place.values.tolist()

    my_location = geocoder.ip('me')

    df_text = dataset['Test Lab Name']

    xlat = dataset['Latitude'].tolist()
    xlon = dataset['Longitude'].tolist()
    locations = list(zip(xlat, xlon))
    map2 = folium.Map(location=my_location.latlng, tiles='CartoDB dark_matter', zoom_start=8)
    marker_cluster = MarkerCluster().add_to(map2)

    title_html = '''
                <style>
                    .button3 {border-radius: 8px;}

                    .button {
                      background-color: #4CAF50; /* Green */
                      border: none;
                      color: white;
                      padding: 15px 32px;
                      text-align: center;
                      margin-left: 42%;
              margin-bottom: 10px;
                      text-decoration: none;
                      display: inline-block;
                      font-size: 16px;
                    }
                </style>
                 <h3 align="center" style="font-size:20px"><b>Covid-19 Active Test Lab In India</b></h3>
                 '''

    map2.get_root().html.add_child(folium.Element(title_html))

    folium.Marker(
                location=my_location.latlng, 
                popup='Me',
                icon=folium.Icon(color='darkblue', icon_color='white', icon='male', angle=0, prefix='fa')
            ).add_to(map2)

    try:
        for point in range(0, len(locations)):
            folium.Marker(locations[point], 
                          popup = folium.Popup(df_text[point]),
                         ).add_to(marker_cluster)    
    except:
        pass
    #map2.save('map.html')

    #bot = Coronavirus()
    #bot.get_data()

    total_case = '343,026'
    total_death = '9,915'
    total_recovere = '180,320'

    return render_template('map.html', total_cases=total_case, total_deaths=total_death, total_recovered=total_recovere)

@app.route("/nearMe/")
def nearMe():
    return render_template('nearMe2.html')

@app.route("/detail/")
def detail():
    return render_template('detail.html')