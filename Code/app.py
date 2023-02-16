from flask import Flask,redirect,url_for,render_template,request, Response
import os
import cv2
import time
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
from video_main import EmotionAnalysisVideo
from emotion_analyzer.media_utils import convert_to_rgb
from emotion_analyzer.emotion_detector import EmotionDetector
from typing import Dict, List
from emotion_analyzer.media_utils import draw_bounding_box_annotation
from emotion_analyzer.media_utils import annotate_warning
from emotion_analyzer.media_utils import annotate_emotion_stats
from emotion_analyzer.media_utils import draw_emoji

app=Flask(__name__)

BASEDIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])

app.config["SECRET_KEY"] = 'adfklasdfkK67986&769row7r1902asdf387132j'
app.config['UPLOAD_PATH'] = os.path.join(BASEDIR, 'static/images')

#初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)#设置摄像头输出宽
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)#设置摄像头输出高
print("start reading video...")
time.sleep(2.0)
print("start working")

emotion_detector = EmotionDetector(
        model_loc="models",
        face_detection_threshold=0.8,
        face_detector="dlib",
    )

def load_emojis(path: str = "data//emoji") -> List:
    emojis = {}

    # list of given emotions
    EMOTIONS = [
        "Angry",
        "Disgusted",
        "Fearful",
        "Happy",
        "Sad",
        "Surprised",
        "Neutral",
    ]

    # store the emoji coreesponding to different emotions
    for _, emotion in enumerate(EMOTIONS):
        emoji_path = os.path.join(path, emotion.lower() + ".png")

        emojis[emotion] = cv2.imread(emoji_path, -1)


    return emojis

emoji_loc = "data"
emoji_path = os.path.join(emoji_loc, EmotionAnalysisVideo.emoji_foldername)
emojis = load_emojis(path=emoji_path)




def annotate_emotion_data(
        emotion_data: List[Dict], image, resize_scale: float
) -> None:


    # draw bounding boxes for each detected person
    for data in emotion_data:
        image = draw_bounding_box_annotation(
            image, data["emotion"], int(1 / resize_scale) * np.array(data["bbox"])
        )

    # If there are more than one person in frame, the emoji can be shown for
    # only one, so show a warning. In case of multiple people the stats are shown
    # for just one person
    WARNING_TEXT = "Warning ! More than one person detected !"

    if len(emotion_data) > 1:
        image = annotate_warning(WARNING_TEXT, image)

    if len(emotion_data) > 0:



        # draw emotion confidence stats
        image = annotate_emotion_stats(emotion_data[0]["confidence_scores"], image)
        # draw the emoji corresponding to the emotion
        image = draw_emoji(emojis[emotion_data[0]["emotion"]], image)

    return image

def gen():
    global gray

    frame_num = 1
    detection_interval = 15
    resize_scale = 0.5
    emotions = None

    while True:
        ret,frame = cap.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = frame

        if frame_num % detection_interval == 0:
            # Scale down the image to increase model
            # inference time.
            smaller_frame = convert_to_rgb(
                cv2.resize(gray, (0, 0), fx=resize_scale, fy=resize_scale)
            )
            # Detect emotion
            emotions = emotion_detector.detect_emotion(smaller_frame)

        if emotions:
            # Annotate the current frame with emotion detection data
            gray = annotate_emotion_data(emotions, gray, resize_scale)

        frame_num += 1


        ret, buffer = cv2.imencode('.jpg', gray)
        gray1 = buffer.tobytes()
        try:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + gray1 + b'\r\n')
        except Exception:
            pass



        
def shotFunc():
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    imgname = f'catpure{timestamp}.jpg'
    print(f'shotFunc is OK, image name :{imgname}')
    if not isinstance(gray, str):
        cv2.imwrite(os.path.join(app.config['UPLOAD_PATH'], imgname), gray)
    return imgname

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_feed2')
# def video_feed2():
#     return Response(gen2(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video():
    capture =  request.args.get('capture')
    if capture:
        imgname = shotFunc()
        return render_template('video.html', imgname=imgname)
    return render_template('video.html')

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#@app.route('/',methods=['GET','POST'])
@app.route('/image.html',methods=['GET','POST'])
def home():
    if request.method=='POST':
        # Handle POST Request here
        return render_template('image.html')
    return render_template('image.html')


@app.route('/video_file')
def video_file():

    return render_template('video_file.html')

@app.route('/detect', methods=['GET'])
def detect():
    if request.method == 'GET':
        imgname = request.args.get('imgname')
        inputtype = request.args.get('type')
        from emotion_analyzer.media_utils import load_image_path

        ob = EmotionAnalysisVideo(
            face_detector="dlib",
            model_loc="models",
            face_detection_threshold=0.0,
        )

        img1 = load_image_path(os.path.join(app.config['UPLOAD_PATH'], imgname))
        emotion, emotion_conf = ob.emotion_detector.detect_facial_emotion(img1)
        print('-'*50)
        print(emotion, emotion_conf)
        if inputtype == "video":
            return render_template('video.html', imgname = imgname, emotion=emotion, emotion_conf=emotion_conf)
        return render_template('image.html', imgname = imgname, emotion=emotion, emotion_conf=emotion_conf)



@app.route('/detect_video', methods=['GET'])
def detect_video():
    if request.method == 'GET':
        imgname = request.args.get('imgname')
        inputtype = request.args.get('type')

        from emotion_analyzer.media_utils import load_image_path
        from emotion_analyzer.media_utils import get_video_writer
        output_path: str = "static/images/what_happen.mp4"

        detection_interval = 15
        resize_scale = 0.5
        video_path = os.path.join(app.config['UPLOAD_PATH'], imgname)


        try:
            cap = cv2.VideoCapture(video_path)
            # To save the video file, get the opencv video writer
            video_writer = get_video_writer(cap, output_path)
            frame_num = 1


            emotions = None

            while True:
                status, frame = cap.read()
                if not status:
                    break

                try:

                    if frame_num % detection_interval == 0:
                        # Scale down the image to increase model
                        # inference time.
                        smaller_frame = convert_to_rgb(
                            cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                        )
                        # Detect emotion
                        emotions = emotion_detector.detect_emotion(smaller_frame)

                    if emotions:
                        # Annotate the current frame with emotion detection data
                        frame = annotate_emotion_data(emotions, frame, resize_scale)


                    video_writer.write(frame)


                except Exception as exc:
                    raise exc
                frame_num += 1


        except Exception as exc:
            raise exc
        finally:
            cap.release()
            video_writer.release()
        # img1 = load_image_path(os.path.join(app.config['UPLOAD_PATH'], imgname))

        # emotion, emotion_conf = ob.emotion_detector.detect_facial_emotion(img1)
        # if inputtype == "video":
        #     return render_template('video.html', imgname = imgname, emotion=emotion, emotion_conf=emotion_conf)
        print(imgname)
        return render_template('video_file.html', imgname = imgname, video_done = True)



@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        filepath = ''
        f = request.files['image']
        if f and allowed_file(f.filename):
            imgname = secure_filename(f.filename)
            filepath = os.path.join(app.config['UPLOAD_PATH'], imgname)
            print('filepath:{}'.format(filepath))
            f.save(filepath)
            
            return render_template("image.html", imgname = imgname)    
        return render_template("image.html")


@app.route('/upload_video', methods=['POST'])
def upload_video():
    if request.method == 'POST':
        filepath = ''
        print(request.files)
        f = request.files['video']
        if f:
            imgname = secure_filename(f.filename)
            filepath = os.path.join(app.config['UPLOAD_PATH'], imgname)
            print('filepath:{}'.format(filepath))
            f.save(filepath)

            return render_template("video_file.html", imgname=imgname)
        return render_template("video_file.html")
#index
@app.route('/')
def index():

    return render_template('index.html')
if __name__ == '__main__':
    #DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=5000,debug=True)
