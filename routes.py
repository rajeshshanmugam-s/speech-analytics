from flask import request, Flask, jsonify
from client import governor
import os

app = Flask(__name__)


@app.route('/uploader')
def audio_transcriber():
    '''
    Gets the file from the local then the Audio file is converted into text.
    Method: GET
    BODY: Form data
    KEY: FILE
    Value: Choose the file
    :return: Transcript of the Audio
    '''
    word_tagging =False
    if not os.path.exists('Audio'):
        os.mkdir('Audio')
    f = request.files['file']
    if 'Word_Tagging' in request.values:
        word_tagging = request.values['Word_Tagging']
    f.save('Audio/'+f.filename)
    text_inference = governor('Audio/' + f.filename, word_tagging)
    return jsonify(text_inference)




if __name__ == '__main__':
    app.run(host= '0.0.0.0' , debug=True, port=5000)