import sys
sys.path.insert(0, 'X_Decoder/')

import os
import logging

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

from PIL import Image
import numpy as np
np.random.seed(1)

import torch
from torchvision import transforms

from X_Decoder.utils.arguments import load_opt_command

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from X_Decoder.xdecoder.BaseModel import BaseModel
from X_Decoder.xdecoder import build_model
from X_Decoder.utils.visualizer import Visualizer
from X_Decoder.utils.distributed import init_distributed
import cv2


logger = logging.getLogger(__name__)

def seg_video(video_pth, transform, model, metadata, output_root):
    # set video input parameters
    video = cv2.VideoCapture(video_pth)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = os.path.basename(video_pth)
    
    # Create Videowriters to generate video output
    #file_ext = ".avi"
    #path_out_vis = os.path.join(output_root, basename.split(".")[0] + file_ext)
    #output_file_vis = cv2.VideoWriter(path_out_vis, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
    #                                    (width, height))
    
    frame_count = 0
    # Processing loop
    while (video.isOpened()):
        # read frame
        ret, frame = video.read()
        if frame is None:
            break
        # predict segmentation with X-DECODER
        with torch.no_grad():
            image_ori = Image.fromarray(frame)
            width = image_ori.size[0]
            height = image_ori.size[1]
            image = transform(image_ori)
            image = np.asarray(image)
            image_ori = np.asarray(image_ori)
            images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

            batch_inputs = [{'image': images, 'height': height, 'width': width}]
            try:
                outputs = model.forward(batch_inputs)
                visual = Visualizer(image_ori, metadata=metadata)

                sem_seg = outputs[-1]['sem_seg'].max(0)[1]
                img_out = visual.draw_sem_seg(sem_seg.cpu(), alpha=0.5) # rgb Image

                if not os.path.exists(output_root):
                    os.makedirs(output_root)
                
                print('Proccessed: frame '+ str(frame_count))
                    
                # write results to video output
                #output_file_vis.write(img_out.get_image())
                img_out.save(os.path.join(output_root, basename.split(".")[0] + '_f' + str(frame_count) + '.png'))
                frame_count = frame_count +1
            except Exception as e:
                print('Failed in predict step: '+ str(e))    
        
def seg_single_im(image_pth, transform, model, metadata, output_root):
    with torch.no_grad():
        image_ori = Image.open(image_pth).convert("RGB")
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        try:
            outputs = model.forward(batch_inputs)
            visual = Visualizer(image_ori, metadata=metadata)

            sem_seg = outputs[-1]['sem_seg'].max(0)[1]
            demo = visual.draw_sem_seg(sem_seg.cpu(), alpha=0.5) # rgb Image

            if not os.path.exists(output_root):
                os.makedirs(output_root)
            demo.save(os.path.join(output_root, 'sem.png'))
        except Exception as e:
            logger.error('Failed in predict step: '+ str(e))  

def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['user_dir'] = absolute_user_dir
    opt = init_distributed(opt)

    # META DATA
    pretrained_pth = os.path.join(opt['WEIGHT'])
    output_root = './output'
    input_pth = 'inputs/agri_0_614.jpeg'

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    stuff_classes = ['soil','leaves','sky']
    stuff_colors = [random_color(rgb=True, maximum=255).astype(int).tolist() for _ in range(len(stuff_classes))]
    stuff_dataset_id_to_contiguous_id = {x:x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes + ["background"], is_eval=True)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(stuff_classes)

    if input_pth.endswith(".jpg") or input_pth.endswith(".png") or input_pth.endswith(".jpeg"):
        seg_single_im(input_pth, transform, model, metadata, output_root)
    else:
        seg_video(input_pth, transform, model, metadata, output_root)


if __name__ == "__main__":
    main()
    sys.exit(0)