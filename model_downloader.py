import os
import tarfile
import urllib.request

def ds_model_downloader():
    if not os.path.exists('models'):
        print('Folder not found')
        url = 'https://github.com/mozilla/DeepSpeech/releases/download/v0.5.1/deepspeech-0.5.1-models.tar.gz'
        file_tmp = urllib.request.urlretrieve(url, filename=None)[0]
        # base_name = os.path.basename(url)
        # file_name, file_extension = os.path.splitext(base_name)
        tar = tarfile.open(file_tmp)
        tar.extractall()
        os.rename('deepspeech-0.5.1-models', 'models')
        return 'Model Downloaded'
    else:
        return "Existing models are there"

