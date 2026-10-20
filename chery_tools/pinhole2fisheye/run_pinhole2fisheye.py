import argparse
import os
import glob

from utils.common import check_input_type, InputType
from utils.pinhole2fisheye import pinhole2fisheye
from inference_realesrgan import super_resolution

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    """图像或视频的针孔转鱼眼
    -i 指定图片路径,视频路径,或图片文件夹路径
    -n 指定模型名称
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input image, video or image folder')
    parser.add_argument('--sr', action='store_true', help='Apply super-resolution.')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x2plus',
        help='Model names: RealESRGAN_x4plus | RealESRGAN_x2plus | realesr-general-x4v3')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument(
        '--fp16', action='store_true', help='Use fp16 precision during inference. Default: fp32.')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')
    parser.add_argument(
        '--kb_coeffs',
        type=float,
        nargs=4,
        default=[0.178562, -0.022853, -0.009897, 0.002929],
        help='Four float values for kb_coeffs.'
    )
    parser.add_argument(
        '-f',
        '--focal_length',
        type=float,
        default=150.0,
        help='Float value for focal length. Smaller values produce stronger fisheye distortion.'
    )
    parser.add_argument('--crop', action='store_true', default=True, help='Crop the black edges of the generated image.')

    args = parser.parse_args()

    input_type = check_input_type(args.input)
    sample_list = []
    fps = -1

    assert input_type in [InputType.IMAGE, InputType.VIDEO, InputType.FOLDER], \
        f"Input path {args.input} is invalid. It should be an image, video or image folder."

    if input_type == InputType.VIDEO:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            raise Exception(f"无法打开视频: {args.input}")

        # 读取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"视频帧率: {fps}, 分辨率: {width}x{height}, 总帧数: {frame_count}")

        frames = []
        with tqdm(total=frame_count, desc="视频针孔转鱼眼进度") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                fisheye_image = pinhole2fisheye(frame, args.focal_length, args.kb_coeffs, crop_valid=args.crop)
                sample_list.append({
                    'imgname': "temp",
                    'extension': ".png",
                    'image': fisheye_image,
                })
                pbar.update(1)

        cap.release()

    if input_type == InputType.IMAGE:
        imgname, extension = os.path.splitext(os.path.basename(args.input))
        image = cv2.imread(args.input)
        fisheye_image = pinhole2fisheye(image, args.focal_length, args.kb_coeffs, crop_valid=args.crop)
        sample_list.append({
            'imgname': imgname,
            'extension': extension,
            'image': fisheye_image,
        })
        print(f"针孔转鱼眼完毕.")

    if input_type == InputType.FOLDER:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
        print(paths)
        for path in tqdm(paths, desc="针孔转鱼眼"):
            imgname, extension = os.path.splitext(os.path.basename(path))
            image = cv2.imread(path)
            fisheye_image = pinhole2fisheye(image, args.focal_length, args.kb_coeffs, crop_valid=args.crop)
            sample_list.append({
                'imgname': imgname,
                'extension': extension,
                'image': fisheye_image,
            })

    if args.sr:
        sample_list = super_resolution(sample_list, args)

    if input_type in [InputType.IMAGE, InputType.FOLDER]:
        for sample in sample_list:
            imgname, extension = sample['imgname'], sample['extension']
            if args.suffix == '':
                save_path = os.path.join(args.output, f'{imgname}{extension}')
            else:
                save_path = os.path.join(args.output, f'{imgname}_{args.suffix}{extension}')
            cv2.imwrite(save_path, sample['image'])

    if input_type == InputType.VIDEO:
        assert fps > 0
        height, width = sample_list[0]['image'].shape[:2]
        imgname, extension = os.path.splitext(os.path.basename(args.input))

        if args.suffix == '':
            save_path = os.path.join(args.output, f'{imgname}{extension}')
        else:
            save_path = os.path.join(args.output, f'{imgname}_{args.suffix}{extension}')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        for sample in tqdm(sample_list, desc="合并保存视频"):
            frame = sample["image"]
            out.write(frame)

        print(f"视频合并完成: {save_path}")


if __name__ == '__main__':
    main()
