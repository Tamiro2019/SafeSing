from flask import Flask, render_template, request, redirect, flash
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename
#from flask_uploads import UploadSet, configure_uploads, patch_request_class
import os

import librosa

app = Flask(__name__)
dropzone = Dropzone(app)

# Config
app.config["AUDIO_UPLOADS"] = "/Users/tamiro/Desktop/Insight/Code/SafeSing/SafeSing_Project/Flask_App/uploads"
app.config["ALLOWED_AUDIO_EXTENSIONS"] = ['WAV']

# Definitions
def allowed_audio(filename):
    if not '.' in filename:
        return False
    ext = filename.rsplit(".",1)[1]
    if ext.upper() in app.config["ALLOWED_AUDIO_EXTENSIONS"]:
        return True
    else:
        return False

def model(wave):

    return 1

# Routes

@app.route('/upload',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        print('posted!')
        if request.files:

            audio = request.files["audio"]

            if audio.filename == "":
                print('image must have a filename')
                return redirect(request.url)

            if not allowed_audio(audio.filename):
                print('file extension must be .wav')
                return redirect(request.url)
            else:
                filename = secure_filename(audio.filename)

                audio.save(os.path.join(app.config["AUDIO_UPLOADS"], filename))

            print('image saved!')

            wave = librosa.load(os.path.join(app.config["AUDIO_UPLOADS"], filename))

            print(wave)

            if model(wave) == 0:
                flash("Your phonation is breathy")
            if model(wave) == 1:
                flash("Your phonation is balanced")
            if model(wave) == 2:
                flash("Your phonation is pressed")

            return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    app.secret_key='12345789'
    app.run(debug=True)


