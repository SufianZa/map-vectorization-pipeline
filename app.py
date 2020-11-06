import uuid
from threading import Thread

from flask import Flask, render_template, abort, Response
from flask import request
import os
import json
import time
import global_variables
from pathlib import Path

from pipeline.run_pipeline import Pipeline

# start flask application
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app._static_folder = os.path.abspath("templates/static/")

# use to clean the dir
pipelinesStack = dict()


def clearDir(uuid):
    del pipelinesStack[uuid]
    for file in global_variables.temp_path.glob('{}_*.*'.format(uuid)):
        os.remove(file)


@app.route('/', methods=['GET'])
def index():
    title = 'Automatische Vektorisierung historischer Katasterkarten'
    index = render_template('layouts/index.html',
                            title=title)
    return index


@app.route('/generate', methods=['POST', 'GET'])
def generate():
    # get data
    data = request.files['file']
    method = int(request.form['close-gaps'])
    sensitivity = float(request.form['sensitivity'])
    douglas = float(request.form['douglas'])
    extract_contours = True if "contours" in request.form and method == 1 else False
    param = float(request.form['param1']) if method == 1 else float(request.form['param2'])

    # print requested parameters
    print('method : ', method, ' parameter :', param,
          ' extract_contours :', extract_contours,
          ' sensitivity :', sensitivity,
          ' douglas :', douglas)

    # generate random uuid
    gen_id = str(uuid.uuid1())
    name, format = data.filename.split('.')

    # check for png format
    if str(format).lower() != 'png':
        abort(Response('The uploaded image should be only in png format '))

    path = os.path.join(global_variables.temp_path, '{}_img.'.format(gen_id) + 'png')
    data.save(path)

    def run_pipeline_in_background(value):
        time.sleep(value)
        pipeline = Pipeline()
        pipelinesStack[gen_id] = pipeline

        pipeline.run_pipeline(path, gen_id, **dict(webapp=True,
                                                   method=method,
                                                   factor=param,
                                                   extract_contour=extract_contours,
                                                   sensitivity=sensitivity,
                                                   douglas=douglas))

    pipelineThread = Thread(target=run_pipeline_in_background, kwargs={'value': request.args.get('value', 1)})
    pipelineThread.start()
    return render_template('layouts/processing.html', uuid=gen_id)


@app.route('/finished', methods=['GET'])
def finished():
    uuid = str(request.args['uuid'])
    if uuid in pipelinesStack:
        msg = pipelinesStack[uuid].get_status()
        if not Path(os.path.join(global_variables.temp_path, uuid + '_simple.geojson')).is_file():
            return {'ok': False, 'msg': msg}
        else:
            return {'ok': True, 'msg': msg}
    else:
        abort(400)


@app.route('/generate/map', methods=['GET'])
def map():
    uuid = str(request.args['uuid'])
    return render_template('layouts/map.html', uuid=uuid)


@app.route('/get_vector', methods=['GET'])
def vector():
    uuid = str(request.args['uuid'])
    with open(Path(global_variables.temp_path, uuid + '_simple.geojson')) as f:
        web = json.load(f)
    with open(Path(global_variables.temp_path, uuid + '_gis.geojson')) as f:
        qgis = json.load(f)
    return {'web': web, 'qgis': qgis}
