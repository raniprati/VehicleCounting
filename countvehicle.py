import cv2
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import os
import shutil
# loading YOLO model
model = YOLO('yolov8x.pt')
# getting all object class names
dict_classes = model.model.names
#print("YOLO classes:", dict_classes)

def resize_frame(frame, scale_percent):
    """Function to resize an image in a percent scale"""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized


def count_vehicle(inputpath, scale_percent):
    video = cv2.VideoCapture(inputpath)
    # Vehicle Objects to detect by Yolo model
    class_IDS = [2, 3, 5, 7]
    veiculos_contador_in = dict.fromkeys(class_IDS, 0)
    contador_in = 0
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video.get(cv2.CAP_PROP_FPS)
    #print('[INFO] - Original Video Dim and FPS: ', (width, height), fps)

    # Scaling Video for better performance
    if scale_percent != 100:
        #print('[INFO] - Scaling change may cause errors in pixels lines ')
        width = int(width * scale_percent / 100)
        height = int(height * scale_percent / 100)
        #print('[INFO] - Dim Scaled Video: ', (width, height))
    video_name = 'result.mp4'
    output_video = cv2.VideoWriter(video_name,
                                   cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                   fps, (width, height))
    success = 1
    while success:
        # vidObj object calls read unction extract frames
        success, frame = video.read()
        if success:
            # Applying resizing of read frame
            frame = resize_frame(frame, scale_percent)
            #print("frame shape: ", frame.shape)
            referenceline = int(3*frame.shape[0] / 4)
            offset = int(fps)
            #print(referenceline, frame.shape[1], offset)
            # Getting predictions from the model
            y_hat = model.predict(frame, conf=0.7, classes=class_IDS, device='cpu', verbose=False)
            classes = y_hat[0].boxes.cls.cpu().numpy()
            # Storing the above information in a dataframe
            positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.boxes,
                                           columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])

            # Translating the numeric class labels to text
            labels = [dict_classes[i] for i in classes]
            # Drawing transition line for Total Vehicles Counting
            cv2.line(frame, (0, referenceline), (frame.shape[1], referenceline), (255, 255, 0), 1)
            for ix, row in enumerate(positions_frame.iterrows()):
                # Getting the coordinates of each vehicle (row)
                xmin, ymin, xmax, ymax, confidence, category, = row[1].astype('int')
                # Calculating the center of the bounding-box
                center_x, center_y = int(((xmax + xmin)) / 2), int((ymax + ymin) / 2)
                # drawing center and bounding-box of vehicle in the given frame
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)  # box
                cv2.circle(frame, (center_x, center_y), 1, (255, 0, 0), -1)  # center of box
                # Drawing above the bounding-box the name of class recognized.
                cv2.putText(img=frame, text=labels[ix],
                            org=(xmin, ymin), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0),
                            thickness=1)
                if (center_y > (referenceline - offset)) and (center_y < referenceline):
                    contador_in += 1
                    veiculos_contador_in[category] += 1
            # updating the counting type of vehicles
            contador_in_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_in.items()]

            # drawing the counting of type of vehicles in the corners of frame
            xt = 10
            cv2.putText(img=frame, text="Total Vehicles: " + str(contador_in),
                    org=(0, xt), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.5, color=(255, 255, 0), thickness=1)
            for txt in range(len(contador_in_plt)):
                xt += 15
                cv2.putText(img=frame, text=contador_in_plt[txt],
                            org=(0, xt), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=0.5, color=(255, 255, 0), thickness=1)
            # saving transformed frames in a output video format
            output_video.write(frame)
    # Releasing the video
    output_video.release()
    print("Total count of vehicles in file: ", inputpath, " :=", contador_in)
    return video_name


if __name__ == '__main__':
    curdir = Path.cwd()
    #print(curdir, type(curdir))
    videodir = Path.joinpath(curdir, "inputdir")
    videolist = os.listdir(videodir)
    outdir = Path.joinpath(curdir, "resultdir")
    shutil.rmtree(outdir, ignore_errors=True)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for vdfile in videolist:
        vdpath = str(Path.joinpath(videodir, vdfile))
        #print(vdpath, type(vdpath))
        # Scaling percentage of original frame
        scale_percent = 70
        outvdfile = count_vehicle(vdpath, scale_percent)
        outpath = Path.joinpath(outdir, str(Path(vdpath).stem) + "_" + str(outvdfile))
        shutil.copyfile(outvdfile, outpath)






