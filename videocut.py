import cv2

path = "../datasets/video/"
target_path = '../datasets/videocut/'
video_name = "VID_0103.MOV"

cap = cv2.VideoCapture(path + video_name)
cap.isOpened()

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)

if cap.isOpened():
    rate = cap.get(5)
    FrameNumber = int(cap.get(7))
    duration = FrameNumber/rate
    fps = int(rate)
    print(rate)
    print(FrameNumber)
    print(duration)
    print(fps)

success, frame = cap.read()
print(success)
print(frame)

i = 0
videoWriter = cv2.VideoWriter(target_path + str(i) + '.mp4', cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (int(width), int(height)))
while (True):
    success, frame = cap.read()
    if success:
        i += 1
        if (i % (5*fps) == 2*fps):
            videoWriter.write(frame)
            videoWriter = cv2.VideoWriter(target_path + str(i) + '.mp4', cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (int(width), int(height)))
        elif (i % (5*fps) < 2*fps):
            videoWriter.write(frame)         
    else:
        print('end')
        break

cap.release()
