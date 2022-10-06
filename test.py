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
    x0 = int((box[0] - ((box[2]) / 2)) * img_width)
    y0 = int((box[1] - ((box[3]) / 2)) * img_height)
    x1 = int((box[0] + ((box[2]) / 2)) * img_width)
    y1 = int((box[1] + ((box[3]) / 2)) * img_height)
    return [x0, y0, x1, y1]

@torch.no_grad()
def load_model(weights="",  # model.pt path(s)
        data='data/coco128.yaml',  # dataset.yaml path
        imgsz=[512, 512],  # inference size (height, width)
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
        imgsz=[512,512],  # inference size (height, width)
        conf_thres=0.1,  # confidence threshold
        iou_thres=0.3,  # NMS IOU threshold
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
                box_image.append(box)
    return box_image

def getMemoryUsage():
    usage = nvsmi.DeviceQuery("memory.used")["gpu"][0]["fb_memory_usage"]
    return "%d %s" % (usage["used"], usage["unit"])



if __name__ == "__main__":
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    # tmp_tensor = torch.empty(int(total_memory * 0.499), dtype=torch.int8, device='cuda')
    folder_img = "/home/anlab/Downloads/薬品複数画像/û≥òiòíÉöëµæ£/"
    folder_output = "/home/anlab/Desktop/Songuyen/PIl_detection/box_output/"
    weight = "/home/anlab/Desktop/Songuyen/PIl_detection/CP/cp_0610.pt"
    
    model, device = load_model(weights=weight)
    print("GPU Memory_____: %s" % getMemoryUsage())

    torch.cuda.set_per_process_memory_fraction(0.5, device)

    for path in os.listdir(folder_img):

        img_ori = cv2.imread(folder_img + path)
        box_img = detect_plate(model, device, img_ori)

        print("GPU Memory: %s" % getMemoryUsage())


