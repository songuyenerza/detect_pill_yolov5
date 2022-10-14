yolo_path = "yolov5"
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
from pynvml.smi import nvidia_smi
nvsmi = nvidia_smi.getInstance()

def convert_box(box, img_width, img_height):
    x0 = int((box[0] - ((box[2]*1.4) / 2)) * img_width)
    y0 = int((box[1] - ((box[3]*1.4) / 2)) * img_height)
    x1 = int((box[0] + ((box[2]*1.4) / 2)) * img_width)
    y1 = int((box[1] + ((box[3]*1.4) / 2)) * img_height)
    return [x0, y0, x1, y1]

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
        imgsz=[640,640],  # inference size (height, width)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.0001,  # NMS IOU threshold
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
    img = letterbox(im0s, imgsz, stride=stride, auto=pt)[0]

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
        # print(len(det))
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            box_image=[]

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                line=(('%g ' * len(line)).rstrip() % line)
                line=line.split(" ")
                line= [float(value) if i!=0 else int(value) for i,value in enumerate(line)]
                cls=line[0]
                box=convert_box(line[1:],im0.shape[1],im0.shape[0])
                if box[0] > int(im0.shape[1]/5):
                    if box[2] < int(im0.shape[1]*0.82):
                        box_image.append(box)
    return box_image
def crop_box(img_ori, box_img,img_output, check_crop):
    img = img_ori
    if len(box_img)!= 0:
        for i in range(len(box_img)):
            croped = img_ori[box_img[i][1]:box_img[i][3], box_img[i][0]: box_img[i][2]]
            croped = cv2.resize(croped,(int(croped.shape[1]*1000/croped.shape[0]), 1000))
            img = cv2.rectangle(img_ori, (box_img[i][0],box_img[i][1]), (box_img[i][2],box_img[i][3]), (0,0,255), 2)
            if check_crop == True:
                cv2.imwrite(img_output , croped)
    return img


def getMemoryUsage():
    usage = nvsmi.DeviceQuery("memory.used")["gpu"][0]["fb_memory_usage"]
    return "%d %s" % (usage["used"], usage["unit"])

if __name__ == "__main__":
    # total_memory = torch.cuda.get_device_properties(0).total_memory
    # tmp_tensor = torch.empty(int(total_memory * 0.499), dtype=torch.int8, device='cuda')
    folder_img = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/Download/train_pilll_1310/train/"
    folder_output = "/home/anlab/Desktop/Songuyen/PIl_detection/check/"
    weight = "/home/anlab/Desktop/Songuyen/PIl_detection/CP/cp_0610v2.pt"
    model, device = load_model(weights=weight)
    print("GPU Memory_____: %s" % getMemoryUsage())
    count = 0
    tt = 0
    for pa in os.listdir(folder_img):
        for pat in os.listdir(folder_img + pa ):
            for path in os.listdir(folder_img + pa + "/" + pat):
                img_ori = cv2.imread(folder_img + pa + "/" + pat + "/" + path)
                center = img_ori.shape
                dis = abs(center[0] - center[1])
                img_ori = img_ori[int(dis/2):int(dis/2 + center[1]), 0: center[1]]
                tt+=1
                box_img = detect_plate(model, device, img_ori,iou_thres = 0.0001)
                img_output = folder_output +  pa + pat +  path[:-4] + 'croped' + str(tt) + ".jpg"
                img = crop_box(img_ori, box_img, img_output, check_crop = False)
    print(tt)