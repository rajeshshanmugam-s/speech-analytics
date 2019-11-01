from flask import request, Flask
from client import transcriber
import os

app = Flask(__name__)


@app.route('/uploader')
def upload_file():
    '''
    Gets the file from the local then the Audio file is converted into text.
    Method: GET
    BODY: Form data
    KEY: FILE
    Value: Choose the file
    :return: File uploaded successfully
    '''
    if not os.path.exists('Audio'):
        os.mkdir('Audio')
    f = request.files['file']
    f.save('Audio/'+f.filename)
    text_inference = transcriber('Audio/' + f.filename)
    return text_inference


# TODO: Mention the Port Number
if __name__ == '__main__':
    app.run(debug=True)
