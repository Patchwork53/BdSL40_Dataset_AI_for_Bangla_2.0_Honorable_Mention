import os
import numpy as np
import cv2
import sys


def get_frames(filename, n_frames= 1):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list= np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frames.append(frame)
    v_cap.release()
    return frames, v_len

def store_frames(frames, path2store):
    for ii, frame in enumerate(frames):
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        path2img = os.path.join(path2store, "frame"+str(ii)+".jpg")
        cv2.imwrite(path2img, frame)

def video_to_images(path2Videos, path2save,n_frames=32):
    for folder in os.listdir(path2Videos):
        os.mkdir(path2save+"/"+folder)
        for video_file in os.listdir(path2Videos+"/"+folder):
            save_to =path2save+"/"+folder+"/"+video_file.split(".")[0] 
            os.mkdir(save_to)
            frames, vlen = get_frames(path2Videos+"/"+folder+"/"+video_file, n_frames)
            store_frames(frames, save_to)


def crop_and_resize_inplace(path2Images, new_dim=100):
    for folder1 in os.listdir(path2Images):
        for vid_folder in os.listdir(path2Images+"/"+folder1):
            for picture in os.listdir(path2Images+"/"+folder1+"/"+vid_folder):
                img_path = path2Images+"/"+folder1+"/"+vid_folder+"/"+picture
                img = cv2.imread(img_path)
                Width, height = img.shape[1], img.shape[0]
                start=int((Width-height)/2) 
                end = int((Width+height)/2)
                img = img[0:height, start:end]
                img = cv2.resize(img, (new_dim, new_dim))
                cv2.imwrite(img_path, img)
  
if __name__ == "__main__":
    n_frames = 32
    new_dim = 100
    path2Videos = sys.argv[1]
    path2save= sys.argv[2]
    video_to_images(path2Videos, path2save, n_frames)
    crop_and_resize_inplace(path2save, new_dim)

    

