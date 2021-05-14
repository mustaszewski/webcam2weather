import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path # do not indluce in requirements
import pandas as pd
import requests
from requests_html import HTMLSession, AsyncHTMLSession
import math # do not include in requirements
import cv2

gpus = tf.config.list_physical_devices('GPU')
#print("Available GPU's for Tensorflow: {}".format(len(gpus)))


def get_screenshots(cam_url, interval=10, first_only=True, get_metadata=True):
    '''Captures a static image of video in pre-defined intervals.
    :param: url String indicating the absolute URL of a mp4 video file
    :param: interval Time interval in seconds (default is 10).
    :return: List of arrays, each of which represents one screenshot
    '''

    metadata = {'location': 'n/a', 'location2':'n/a', 'time':'n/a', 'temp':'n/a', 'wind':'n/a'}
    # start html session
    session = HTMLSession()
    page = session.get(cam_url)
    url = page.html.find('source')[0].attrs.get('src')
    loc = page.html.find('div.playercaption_txt')
    loc = loc[0].text if len(loc)>0 else "n/a"
    metadata["location"] = loc
    loc2 = page.html.find('img.cams_img_a')
    loc2 = loc2[0].attrs.get('alt').split(',')[0] if len(loc2)>0 else "n/a"
    metadata['location2'] = loc2
    time = page.html.find('div#video_clock_div') # page.html.find('div#video_clock_div')[0].text
    time = time[0].text if len(time) > 0 else "n/a"
    metadata['time'] = time
    temp = page.html.find('div#video_wetter_vp_div') # page.html.find('div#video_wetter_vp_div')[0].text
    temp = temp[0].text if len(temp) > 0 else "n/a"
    metadata['temp'] = temp
    wind = page.html.find('input#hidden_wetterWind_div')
    wind = wind[0].attrs.get('value') if len(wind) > 0 else "n/a"
    metadata['wind'] = wind

    screenshots = []
    # capture video
    vidcap = cv2.VideoCapture(url)
    success,image = vidcap.read()

    if first_only:
        screenshots.append(image)
    else:

        fps = vidcap.get(cv2.CAP_PROP_FPS) # get the frames per second
        multiplier = fps * interval
    
        while success:
            frameId = int(round(vidcap.get(1))) #current frame number, rounded b/c sometimes you get non-integer frame intervals
            success, image = vidcap.read()

            if frameId % multiplier == 0:
                screenshots.append(image)

    vidcap.release()
    return (screenshots, metadata)

def process_image(image, IMG_WIDTH=160, IMG_HEIGHT=160):
    #image = image.resize([IMG_WIDTH, IMG_HEIGHT])
    image = cv2.resize(image, dsize=(IMG_WIDTH, IMG_HEIGHT))
    img = tf.convert_to_tensor(np.array(image))
    img = tf.cast(img, tf.float32)
    return tf.expand_dims(img, 0)


model_dir = Path.cwd() / "models" 

# dict of cam names and cam urls
cams = {'Axamer Lizum - Hoadl' : 'https://webtv.feratel.com/webtv/?cam=5510',\
'Axamer Lizum - Birgitzköpflhaus':'https://webtv.feratel.com/webtv/?cam=5511',\
'Seegrube' : 'https://webtv.feratel.com/webtv/?cam=5645',\
'Hungerburg' : 'https://webtv.feratel.com/webtv/?cam=5646',\
'Schlick - Krinnenkopf':'https://webtv.feratel.com/webtv/?cam=5670',\
'Schlick - Mitterjoch':'https://webtv.feratel.com/webtv/?cam=5671',\
'Fügen - Onkeljoch':'https://webtv.feratel.com/webtv/?cam=5545',\
'Hintertux - Rastkogel':'https://webtv.feratel.com/webtv/?cam=5554',\
'Hochötz - Wetterkreuzbahn':'https://webtv.feratel.com/webtv/?cam=5756',\
'Hochötz - Widiversum':'https://webtv.feratel.com/webtv/?cam=5757',\
'Achenkirch - Christlum*':'https://webtv.feratel.com/webtv/?cam=5501',\
'Weerberg - Hütegglift*':'https://webtv.feratel.com/webtv/?cam=5783',\
'Pillberg - Kellerjochbahn*':'https://webtv.feratel.com/webtv/?cam=5780',\
'Stubaier Gletscher - Daunjoch*':'https://webtv.feratel.com/webtv/?cam=5711',\
'Matrei am Brenner - Maria Waldrast*':'https://webtv.feratel.com/webtv/?cam=5747',\
'Kolsassberg - Hoferlift*':'https://webtv.feratel.com/webtv/?cam=5785',\
'Nordkette - Hafelekar*':'https://webtv.feratel.com/webtv/?cam=5647',\
'Galtür - Breitspitzbahn*':'https://webtv.feratel.com/webtv/?cam=5547',\
'Haiming - Ochsengarten*':'https://webtv.feratel.com/webtv/?cam=5758',\
'Hintertuxer Gletscher - Gefrorene Wand*':'https://webtv.feratel.com/webtv/?cam=5552',\
'Kitzbühel - Hahnenkamm*':'https://webtv.feratel.com/webtv/?cam=5604',\
'Fieberbrunn - Streuböden Mittelstation*':'https://webtv.feratel.com/webtv/?cam=5532',\
'Mayrhofen - Penkenbahn*':'https://webtv.feratel.com/webtv/?cam=5638',\
}
        
cam_names = list(cams.keys())
help_camselector = "Cams marked with an asterisk * were not represented in the training data, thus predictions may be less accurate."

# labels
class_names = ['clear', 'cloudy_mostly', 'cloudy_partially', 'foggy', 'rainy', 'snowy']

IMG_SIZE = 160


st.title("Webcam2Weather")


model_name = "mnet_v2_weights.h5"
model = tf.keras.models.load_model(model_dir / model_name) # "mnet_v2_weights.h5"
#uploaded_file = "img_test_rain.jpg"

selector_cam = st.selectbox("Choose webcam", options=cam_names, help=help_camselector)

help_fetch_multiple = "If set to 'yes', predictions are averaged across multiple screenshots taken from the webcam in intervals of 10 seconds, thus providing a more reliable prediction.\
    However, it slows down the app, b/c the entire video stream rather than the first video frame needs to be fetched."
fetch_multiple = st.radio("Fetch multiple screenshots?", ("yes", "no"), index=1, help=help_fetch_multiple)

first_only = True if fetch_multiple == "no" else False
cam_url = cams[selector_cam]

screenshots, metadata = get_screenshots(cam_url, first_only=first_only)
screenshots_tf = [process_image(img) for img in screenshots]

if len(screenshots) > 0:
    image = screenshots[0]
    img = screenshots_tf[0]
    #image = Image.open(uploaded_file)
    #img = process_image(image)
    caption = "Latest webcam stream from {} ({})".format(selector_cam.strip("*"), cam_url)
    st.write(caption, )

    st.image(screenshots, use_column_width=True, channels="bgr") #caption=caption, 
    #st.write("Time: {}\tTemp: {}\tWind: {}".format(metadata['time'], metadata['temp'], metadata['wind']))
    pred_probabilities = [model.predict(img) for img in screenshots_tf]
    pred_probabilities_stacked = np.stack(pred_probabilities, axis=0)
    pred_probabilities_mean = np.mean(pred_probabilities_stacked, axis=0)
    #pred_probabilities = model.predict(img)
    max_probability = np.round(np.max(pred_probabilities_mean)*100, 2)
    pred_label = class_names[np.argmax(pred_probabilities_mean)]
    #proba = model.predict(img)[0][0]
    st.write("Predicted weather: {} ({}% probability)".format(pred_label, max_probability))

    chart_data = pd.DataFrame(pred_probabilities_mean.reshape(6,), index=class_names, columns=np.array(["class probability"]))
    #bar_plot = plt.bar(height=pred_probabilities.reshape(6,), x = class_names)

    st.bar_chart(chart_data)
    #st.bar_chart(pred_probabilities.reshape(6,))
    #st.write("%.0f%% sure that person wears a mask" % (proba * 100))

    st.write("Latest weather data from webcam:")
    metadata_table = pd.DataFrame(np.array([metadata['time'], metadata['temp'], metadata['wind']]).reshape(1,3), \
        columns=list(metadata.keys())[-3:], index=['latest data'])
    st.table(metadata_table)

