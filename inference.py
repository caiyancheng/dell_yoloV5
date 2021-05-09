import argparse
import time
import sys


def timer(func):
    def wrapper(*args, **kw):
        t1 = time.time()
        func(*args, **kw)
        t2 = time.time()
        dt = t2 - t1
        print('Execution time: {}'.format(dt))

    return wrapper


parser = argparse.ArgumentParser(
    description='Speed limit object detection',
    prog='inference.py')
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


@timer
def inference(t):
    """
    for imgs in os.list(input_dir):

    """
    t.predict()


def _main(args):
    """
    description:
      this project aims to test the accuracy and efficiency of the model
      ....
      ....

    input:
      model_path:xxxxxx
      input_dir:
      output_dir
    output:
      a txt file which follows xxx format:
          xxxx.jpg #ofObj idx0 x0 y0 dx0 dy0 idx1 x1 y1 dx1 dy1....
          .....
    """
    model_path = args.model_path
    input_dir = args.input_dir
    output_dir = args.output_dir

    sys.path.append(model_path)
    import interface
    t = interface.yolo_model()
    t.init_predict(input_dir, output_dir)
    inference(t)
    t.output()


if __name__ == '__main__':
    args = parser.parse_args()
    _main(args=args)

"""
python inference.py --model_path=xxx --input_dir=xxx --output_dir=xxx like:
python inference.py --model_path=path/dell_yoloV5 --input_dir=image_dir_path --output_dir=output_path
"""
