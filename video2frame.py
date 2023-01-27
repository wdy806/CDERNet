import cv2
import os

videos_path = 'datasets/video/'
name = '3175'
frames_save_path = 'datasets/video/frames/'
time_interval = 10

def video2frame(videos_name, frames_save_path, time_interval):
    if (os.path.exists(frames_save_path) != 1):
        os.mkdir(frames_save_path)
    vidcap = cv2.VideoCapture(videos_name)
    count = 0
    success, image = vidcap.read()
    cv2.imencode('.jpg', image)[1].tofile(frames_save_path + "frame%d.jpg" % count)
    while success:
        success, image = vidcap.read()
        count += 1
        if count % time_interval == 0:
            cv2.imencode('.jpg', image)[1].tofile(frames_save_path + "frame%d.jpg" % count)
    print(count)

video2frame(videos_path  + name + '.mp4', frames_save_path  + name + '/', time_interval)