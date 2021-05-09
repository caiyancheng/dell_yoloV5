import argparse
from pathlib import Path
import torch
import yaml
from tqdm import tqdm
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import check_dataset, check_img_size, \
    non_max_suppression, scale_coords, colorstr
from utils.torch_utils import select_device


class yolo_model:
    def __init__(self):
        self.data = ""
        self.imgs = []
        self.shapes_list = []
        self.paths_list = []
        self.out = []

    def generate_yaml(self, input: str):
        with open('test.yaml', 'w', encoding='utf-8') as f:
            f.write("val: " + input + "\n")
            f.write("nc: 2\n\nnames: ['cancel','limit']")
        self.data = "test.yaml"

    def opts_set(self):
        parser = argparse.ArgumentParser(prog='interface.py')
        parser.add_argument('--weights', nargs='+', type=str, default='./runs/train_crossdomain/exp26/weights/best.pt',
                            # 7
                            help='model.pt path(s)')
        parser.add_argument('--data', type=str, default='data/dell.yaml', help='*.data path')
        parser.add_argument('--batch-size', type=int, default=84, help='size of each image batch')
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--task', default='val', help='train, val, test, speed or study')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--verbose', action='store_true', help='report mAP by class')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
        parser.add_argument('--project', default='runs/test', help='save to project/name')
        parser.add_argument('--name', default='exp', help='save to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument(
            '-m',
            '--model_path',
            type=str,
            default='',
            help='path to trained model')
        parser.add_argument(
            '-i',
            '--input_dir',
            type=str,
            default='',
            help='path to input image folder')
        parser.add_argument(
            '-o',
            '--output_dir',
            type=str,
            default='',
            help='output txt directory')
        self.opt = parser.parse_args()

    def load_model(self):
        self.device = select_device(self.opt.device, batch_size=self.opt.batch_size)

        # Load model
        self.model = attempt_load(self.opt.weights, map_location=self.device)  # load FP32 model
        self.model.half()
        self.gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(self.opt.img_size, s=self.gs)  # check img_size

    def load_dataset(self):
        # Configure
        self.model.eval()
        with open(self.data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        check_dataset(data)  # check
        self.nc = int(data['nc'])  # number of classes

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.opt.img_size, self.opt.img_size).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        self.dataloader = \
            create_dataloader(data[self.opt.task], self.opt.img_size, self.opt.batch_size, self.gs, self.opt, pad=0.5,
                              rect=True,
                              prefix=colorstr(f'{self.opt.task}: '))[0]

    def init_predict(self, input_dir, output_dir: str):
        if (output_dir.endswith("/")):
            self.output_dir = output_dir[0:-1]
        else:
            self.output_dir = output_dir
        self.opts_set()
        self.generate_yaml(input_dir)
        self.load_model()
        self.load_dataset()
        self.seen = 0
        self.names = {k: v for k, v in
                      enumerate(self.model.names if hasattr(self.model, 'names') else self.model.module.names)}
        for index, (img, _, paths, shapes) in enumerate(tqdm(self.dataloader)):
            img = img.to(self.device, non_blocking=True)
            img = img.half()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = img.shape
            self.imgs.append(img)
            self.paths_list.append(paths)
            self.shapes_list.append(shapes)

    def output(self):
        for i in range(len(self.imgs)):
            img = self.imgs[i]
            paths = self.paths_list[i]
            shapes = self.shapes_list[i]
            lb = []
            out = self.out[i]
            out = non_max_suppression(out, conf_thres=self.opt.conf_thres, iou_thres=self.opt.iou_thres, labels=lb,
                                      classes=None,
                                      multi_label=False)

            for si, pred in enumerate(out):
                self.seen += 1
                path = Path(paths[si])
                if len(pred) == 0:
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                for *xyxy, conf, cls in predn.tolist():
                    xyxy_out = torch.tensor(xyxy).view(-1).tolist()  # normalized xywh
                    line = (cls, *xyxy_out)  # label format
                    with open(self.output_dir + '/result.txt', 'a') as f:
                        f.write(path.stem + ".jpg " + ('%g ' * len(line)).rstrip() % line + '\n')

    def predict(self):
        for i in range(len(self.imgs)):
            img = self.imgs[i]
            # with torch.no_grad():
                # Run model
            out, _ = self.model(img, augment=False)  # inference and training outputs

            self.out.append(out)
