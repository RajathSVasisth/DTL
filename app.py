from flask import Flask, render_template, redirect, url_for, flash, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm 
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy  import SQLAlchemy
import aes_ecc_hybrid
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
#pertaining to detection
import numpy as np
import os
import time
import shutil
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_CKPT = "C:/Users/rajat/OneDrive/Desktop/DTLFinal/frozen_inference_graph.pb"
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'C:/Users/rajat/OneDrive/Desktop/DTLFinal/label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 6
#detection

#detection
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile("C:/Users/rajat/OneDrive/Desktop/DTLFinal/frozen_inference_graph.pb", 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
#detection

UPLOAD_FOLDER = 'C:/Users/rajat/OneDrive/Desktop/DTLFinal/static'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Thisissupposedtobesecret!'
app.config["CACHE_TYPE"] = "null"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/rajat/OneDrive/Desktop/DTLFinal/database.db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')

class RegisterForm(FlaskForm):
    email = StringField('email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            decrypt_password = aes_ecc_hybrid.decrypter(user.password)
            if check_password_hash(decrypt_password, form.password.data):
                print('hi')
                login_user(user, remember=form.remember.data)
                return redirect(url_for('upload_image'))

        return '<h1>Invalid username or password</h1>'
        #return '<h1>' + form.username.data + ' ' + form.password.data + '</h1>'

    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        Crypt_password = aes_ecc_hybrid.arguement_taker(hashed_password)
        new_user = User(username=form.username.data, email=form.email.data, password=Crypt_password)
        db.session.add(new_user)
        db.session.commit()

        return '<h1>New user has been created!</h1>'
        #return '<h1>' + form.username.data + ' ' + form.email.data + ' ' + form.password.data + '</h1>'

    return render_template('signup.html', form=form)

@app.route('/upload_image', methods=['GET', 'POST'])
@login_required
def upload_image():
    if request.method == 'POST':
        folder = 'C:/Users/rajat/OneDrive/Desktop/DTLFinal/static'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if 'Input' in str(filename) or 'Output' in str(filename) or 'File' in str(filename):
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            except Exception as e:
                continue
        if 'file' not in request.files:
            #flash('No file part')
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            #flash('No selected file')
            return 'No selected file'
        if file and allowed_file(file.filename):
            input_name = 'Input' + str(time.time()) + '.jpg'
            file.filename = input_name
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #flash('success')
            with detection_graph.as_default():
                with tf.compat.v1.Session(graph=detection_graph) as sess:
                    image_np = cv2.imread(os.path.join("C:/Users/rajat/OneDrive/Desktop/DTLFinal/static",input_name))
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
                    l={}
                    for i in range(len(classes[0])):
                        if scores[0][i]>0.5:
                            st=category_index.get(classes[0][i])['name']
                            if st in l.keys():
                                l[st]+=1
                            else:
                                l[st]=1
                    save_path = 'C:/Users/rajat/OneDrive/Desktop/DTLFinal/static/'
                    name='File'+str(time.time())+'.txt'     
                    f = open(os.path.join(save_path, name), "x")
                    for key, value in l.items():
                        f.write('%s:%s\n' % (key, value))
                    f.close()
                    vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8)
                    # Display output
                    patho = 'C:/Users/rajat/OneDrive/Desktop/DTLFinal/static'
                    output_name = 'Output'+str(time.time())+'.png'
                    cv2.imwrite(os.path.join(patho , output_name), image_np)
                    return render_template('outcome.html', Input=input_name, Output=output_name,File=name)
    else:
        return render_template('upload_image.html', name=current_user.username)

@app.route('/outcome',methods=['GET','POST'])
@login_required
def outcome():
    if request.method == 'POST':
        return render_template('upload_image.html', name=current_user.username)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
