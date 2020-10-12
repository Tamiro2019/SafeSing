from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
from Models import model_prediction
from Models import load_model
from Plotter import make_plot_wave
from Plotter import make_plot_pitches
import numpy as np
import librosa
import os
import webbrowser

application = Flask(__name__)
application.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

cwd = os.getcwd()
# Config
application.config["AUDIO_UPLOADS"] = cwd+'/static/uploads' #cwd+'/uploads' #"/Users/tamiro/Desktop/Insight/Code/SafeSing/SafeSing_Project/Flask_App/uploads"
application.config["ALLOWED_AUDIO_EXTENSIONS"] = ['WAV']

path_uploads = application.config["AUDIO_UPLOADS"]

# Load Model
model = load_model()

# Definitions
def allowed_audio(filename):
    if not '.' in filename:
        return False
    ext = filename.rsplit(".",1)[1]
    if ext.upper() in application.config["ALLOWED_AUDIO_EXTENSIONS"]:
        return True
    else:
        return False


def analyze_wave(model, wave, path):

    print(len(wave))

    if len(wave) < int(2*44100):

        image_path = make_plot_wave(wave, path)

        wave_pred = model_prediction(model, wave)

        if wave_pred == 0:
            flash("Your phonation is breathy")
        if wave_pred == 1:
            flash("Your phonation is balanced")
        if wave_pred == 2:
            flash("Your phonation is pressed")

    else:

        chunk_size = 22050
        class_array = np.zeros(int(np.ceil(len(wave) // chunk_size)), dtype=int)
        chunk_lab = np.zeros(int(np.ceil(len(wave) // chunk_size)), dtype=int)

        for chunk in range(len(chunk_lab)):
            # within this bucket, compare the average pitch to the start and end pitches
            # if the standard deviation is within a quarter tone of the mean, then good

            wavelet = wave[chunk * chunk_size:(chunk + 1) * chunk_size]

            f0_chunk, voiced_flag, voiced_probs = librosa.pyin(y=wavelet, sr=44100, fmin=librosa.note_to_hz('C3'),fmax=librosa.note_to_hz('C5'))
            avg = np.nanmean(f0_chunk)
            std = np.nanstd(f0_chunk)

            if np.count_nonzero(voiced_flag == False) > 8:
                chunk_lab[chunk] = 0 # not voiced
            else:
                if (avg * (35 / 36) < avg - std) and (avg * (36 / 35) > avg + std):  # within a semitone
                    # classify this piece
                    chunk_lab[chunk] = 2 # voiced and stable pitch
                    class_array[chunk] = model_prediction(model, wavelet)

                else:
                    # don't classify
                    chunk_lab[chunk] = 1 # voiced and pitch change

        image_path = make_plot_pitches(wave, chunk_lab, chunk_size, class_array, path)

    return image_path

    # Routes

@application.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print('posted!')
        if request.form.get("breathy", False):
            webbrowser.open_new_tab('https://www.youtube.com/results?search_query=sing+with+without+breathy')
            return redirect(request.url)
        if request.form.get("balanced", False):
            webbrowser.open_new_tab('https://www.youtube.com/results?search_query=sing+with+flow+phonation')
            return redirect(request.url)
        if request.form.get("pressed", False):
            webbrowser.open_new_tab('https://www.youtube.com/results?search_query=remove+tension+singing')
            return redirect(request.url)

        if request.files:
            #print(request.form)

            if request.form.get("example", False):

                #filename = 'example.wav'
                filename = 'example_long.wav'
                wave = librosa.load(os.path.join(application.config["AUDIO_UPLOADS"], filename), sr=44100)[0]

                image_path = 'pitch_plot.svg' #analyze_wave(model, wave, path_uploads)
                print( 'image_path', image_path)
                return render_template('results.html', wave_file=filename, image_file=image_path)

            audio = request.files["audio"]

            if audio.filename == "":
                print('image must have a filename')
                return redirect(request.url)

            if not allowed_audio(audio.filename):
                print('file extension must be .wav')
                return redirect(request.url)
            else:
                filename = secure_filename(audio.filename)

                audio.save(os.path.join(application.config["AUDIO_UPLOADS"], filename))
                print('image saved!')

            # load with librosa and run through model
            wave = librosa.load(os.path.join(application.config["AUDIO_UPLOADS"], filename), sr=44100)[0]

            image_path = analyze_wave(model, wave, path_uploads)

            return render_template('results.html', wave_file=filename, image_file=image_path) #redirect(request.url)

    return render_template('index.html')



# No cacheing at all for API endpoints.
@application.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response

if __name__ == '__main__':

    application.secret_key = '12345789'
    application.run(port=5000, debug=True)
    #application.run(port=8000,host='0.0.0.0',debug=True)


