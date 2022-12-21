yolo_path = "yolov5"
import shutil
import sys
from turtle import window_height
import cv2
import os
import numpy as np
sys.path.append(yolo_path)
from utils.augmentations import letterbox
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, time_sync
from models.common import DetectMultiBackend
import torch
import torch.backends.cudnn as cudnn
import timeit
from pynvml.smi import nvidia_smi
nvsmi = nvidia_smi.getInstance()

def convert_box(box, img_width, img_height):
    x0 = int((box[0] - ((box[2]) / 2)*0.9) * img_width)
    y0 = int((box[1] - ((box[3]) / 2)*0.9) * img_height)
    x1 = int((box[0] + ((box[2]) / 2)*0.9) * img_width)
    y1 = int((box[1] + ((box[3]) / 2)*0.9) * img_height)
    if x0<0:
        x0 = 0
    if y0<0:
        y0 = 0
    return [x0, y0, x1, y1]

def letterbox_edit(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    h, w = im.shape[:2]
    max_wh = max(h, w)
    pad_top = int((max_wh - h)/2)
    pad_left = int((max_wh - w)/2)
    im = cv2.copyMakeBorder(im, pad_top, pad_top, pad_left, pad_left, cv2.BORDER_CONSTANT, value=color) 
    # cv2.imwrite("/home/anlab/Desktop/Songuyen/PIl_detection/check5.png", im)
    return im, ratio, (dw, dh)

@torch.no_grad()
def load_model(weights="",  # model.pt path(s)
        data='data/coco128.yaml',  # dataset.yaml path
        imgsz=[640, 640],  # inference size (height, width)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    model.warmup(imgsz=(1 , 3, *imgsz))  # warmup
    # print("device",device)
    return model,device
@torch.no_grad()

def detect_plate(model,
        device,
        source,  # file/dir/URL/glob, 0 for webcam
        imgsz=[736,736],  # inference size (height, width)
        conf_thres=0.35,  # confidence threshold
        iou_thres=0.1,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        ):
    
    # Load model
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    im0s = source
    img = letterbox_edit(im0s, imgsz, stride=stride, auto=pt)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(img)

    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im)
    im0s = source
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    
    result=[]
    # Process predictions
    for i, det in enumerate(pred):  # per image
        im0= im0s.copy()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        box_image=[]
        list_line_save = []

        # print(len(det))
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            box_image=[]
            list_line_save = []
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                line_save = line
                list_line_save.append(line_save)
                line=(('%g ' * len(line)).rstrip() % line)
                line=line.split(" ")
                line= [float(value) if i!=0 else int(value) for i,value in enumerate(line)]
                cls=line[0]
                box=convert_box(line[1:],im0.shape[1],im0.shape[0])
                # if box[0] > int(im0.shape[1]*0.02):
                # if int(im0.shape[1]*0.1) < box[2] < int(im0.shape[1]*0.82):
                box_image.append(box)
    return box_image, list_line_save

def crop_box(img_ori, box_img,img_output, check_crop):
    img = img_ori
    img_orii = img_ori.copy()
    if len(box_img)!= 0:
        crop_list = []
        for i in range(len(box_img)):
            croped = img_ori[box_img[i][1]:box_img[i][3], box_img[i][0]: box_img[i][2]]
            # try:
            #     croped = cv2.resize(croped,(int(croped.shape[1]*300/croped.shape[0]), 300))
            # except:
            #     print(croped.shape)
            #     print(box_img[i])
            crop_list.append(croped)
            img = cv2.rectangle(img_orii, (box_img[i][0],box_img[i][1]), (box_img[i][2],box_img[i][3]), (0,0,255), 2)
            if check_crop == True:
                cv2.imwrite(img_output , croped)
    if check_crop == True:
        return crop_list
    else:
        return img

def getMemoryUsage():
    usage = nvsmi.DeviceQuery("memory.used")["gpu"][0]["fb_memory_usage"]
    return "%d %s" % (usage["used"], usage["unit"])

if __name__ == "__main__":
    folder_img = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/pill_detect/test1912/test/"
    folder_output = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/pill_detect/false/"
    output_label = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/pill_detect/label_false/"
    weight = "/home/anlab/Desktop/Songuyen/PIl_detection/CP/cp2112_best.pt"
    model, device = load_model(weights=weight)
    print("GPU Memory_____: %s" % getMemoryUsage())
    count = 0
    tt = 0
    err = 0
    data_folder = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/pill_detect/test1912/test/"
    with open( data_folder +  'paths.txt','r') as f:
        IMAGE_PATH_DB = [line.strip('\n') for line in f.readlines()]


    for i in range(len(IMAGE_PATH_DB)):
        count += 1
        start = timeit.default_timer()
        # for pat in os.listdir(folder_img + pa ):
            # for path in os.listdir(folder_img + pa + "/" + pat):
        img_input = folder_img  + IMAGE_PATH_DB[i]
        img_ori = cv2.imread(folder_img  + IMAGE_PATH_DB[i])
        img_output = folder_output +  IMAGE_PATH_DB[i]
        save_txt = output_label + IMAGE_PATH_DB[i].split('/')[-1][:-4]
        center = img_ori.shape
        tt+=1
        box_img, list_line_save = detect_plate(model, device, img_ori,imgsz=[640,640],conf_thres=0.4, iou_thres = 0.3)

        # img_out = crop_box(img_ori, box_img, img_output, check_crop = False)  #check == True --> croped_list, check==False ---> img_rectangle
        # cv2.imwrite( img_output, img_out)

    #     print(timeit.default_timer() -start)
        # if len(box_img) == 0:
        #     w =  center[1] * 0.8
        #     h =  center[0] * 0.8
        #     x = center[1]/2 - w/2
        #     y = center[0]/2 - h/2
        #     img_ori = img_ori[int(y):int(y+h), int(x):int(x+w)]
        #     box_img, list_line_save = detect_plate(model, device,img_ori, imgsz=[704,704],conf_thres = 0.6 ,iou_thres = 0.1)
        if len(box_img) % 5 != 0:
            print(len(box_img), img_input)
            # shutil.copy(img_input, folder_output)
            # for line in list_line_save:
            #     with open(f'{save_txt}.txt', 'a') as f:
            #         f.write(('%g ' * len(line)).rstrip() % line + '\n')
        else:
            err+=1
            print(err)
    #         img_out = crop_box(img_ori, box_img, img_output, check_crop = False)  #check == True --> croped_list, check==False ---> img_rectangle
    #         # cv2.imwrite( img_output, img_out)
    #     else:
    #         img_out = crop_box(img_ori, box_img, img_output, check_crop = False)  #check == True --> croped_list, check==False ---> img_rectangle
    #         # cv2.imwrite( img_output, img_out)
    #         print(img_output)
    #         print(len(box_img))
    #         err +=1
    #         # for i in range(len(img_out)):
    #         #     count+=1
    #         #     cv2.imwrite("/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/pill_detect/data_train2012/" + path, img_out[i])
    # print(err, count, err/count)
    print(err)