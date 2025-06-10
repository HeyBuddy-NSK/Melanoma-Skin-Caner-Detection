from flask import Flask, render_template, request, redirect, url_for
from skin_detect import SkinDetect
import os

app = Flask(__name__)

## Creating upload folder for saving uploaded images
path = os.getcwd()

UPLOAD_FOLDER = os.path.join(path, 'uploads\\')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
    
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


## Code for user to allow only specific types of file.
ALLOWED_EXTENSIONS = ['png','jpg','jpeg']

def allowed_file(filename):
    """
    Function to chek if file is image of not
    """
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """
    Function to render main page
    """
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():

    """
    Function to process the image and detect.
    """

    detect_canc = SkinDetect()
    
    if 'image' not in request.files:
        return redirect(url_for('index'))

    image = request.files['image']

    if image.filename == '':
        return redirect(url_for('index'))
    
    if image and allowed_file(image.filename):
            image.save(os.path.join(app.config['UPLOAD_FOLDER'],image.filename))
            
            img_src = os.path.join(app.config['UPLOAD_FOLDER'],image.filename)
            
            result_answer, confidence = detect_canc.detection(img_src)
            
            return render_template("result.html", answer = result_answer,
                                   confidence = confidence)

    else:
        return render_template('result.html', result_image=result_answer)

if __name__ == '__main__':
    app.run(debug=True)
