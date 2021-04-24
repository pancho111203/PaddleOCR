# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import pandas as pd
import numpy as np
import time
import sys
from tqdm.auto import tqdm
import fire
import tools.infer.utility as utility
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process

logger = get_logger()

DATA_FOLDER = os.environ.get(
    'DATA_FOLDER', '/home/frubin/Projects/Kaggle/shopee-product-matching/data'
)

class TextE2E(object):
    def __init__(self, args):
        self.args = args
        self.e2e_algorithm = args.e2e_algorithm
        pre_process_list = [{
            'E2EResizeForTest': {}
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {}
        if self.e2e_algorithm == "PGNet":
            pre_process_list[0] = {
                'E2EResizeForTest': {
                    'max_side_len': args.e2e_limit_side_len,
                    'valid_set': 'totaltext'
                }
            }
            postprocess_params['name'] = 'PGPostProcess'
            postprocess_params["score_thresh"] = args.e2e_pgnet_score_thresh
            postprocess_params["character_dict_path"] = args.e2e_char_dict_path
            postprocess_params["valid_set"] = args.e2e_pgnet_valid_set
            postprocess_params["mode"] = args.e2e_pgnet_mode
            self.e2e_pgnet_polygon = args.e2e_pgnet_polygon
        else:
            logger.info("unknown e2e_algorithm:{}".format(self.e2e_algorithm))
            sys.exit(0)

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors = utility.create_predictor(
            args, 'e2e', logger)  # paddle.jit.load(args.det_model_dir)
        # self.predictor.eval()

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, images):
        start_time = time.time()
        imgs = []
        shape_lists = []
        for img in images:
            ori_im = img.copy()
            data = {'image': img}
            data = transform(data, self.preprocess_op)
            imgs.append(data[0])
            shape_lists.append(data[1])

        img = np.stack(imgs)
        shape_list = np.stack(shape_lists)
        img = img.copy()

        self.input_tensor.copy_from_cpu(img)

        preprocess_time = time.time()
        self.predictor.run()

        model_infer_time = time.time()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)

        preds = {}
        if self.e2e_algorithm == 'PGNet':
            preds['f_border'] = outputs[0]
            preds['f_char'] = outputs[1]
            preds['f_direction'] = outputs[2]
            preds['f_score'] = outputs[3]
        else:
            raise NotImplementedError
        post_result = self.postprocess_op(preds, shape_list)
        postprocess_time = time.time()
        dt_boxes_all = []
        strs_all = []
        mean_scores_all = []
        min_scores_all = []
        for res in post_result: 
            points, strs, mean_scores, min_scores = res['points'], res['texts'], res['mean_scores'], res['min_scores']
            dt_boxes = self.filter_tag_det_res_only_clip(points, ori_im.shape)

            dt_boxes_all.append(dt_boxes)
            strs_all.append(strs)
            mean_scores_all.append(mean_scores)
            min_scores_all.append(min_scores)

        print(f'Preprocess time = {preprocess_time - start_time}')
        print(f'Infer time = {model_infer_time - preprocess_time}')
        print(f'Postprocess time = {postprocess_time - model_infer_time}')
        elapse = time.time() - start_time
        return dt_boxes_all, strs_all, mean_scores_all, min_scores_all, elapse


def predict_shopee_ds(args, images_path='train_images', dataframe_path='train.csv', batch_size=32):
    # Try with different values of:
        # args.gpu_mem
        # args.use_tensorrt:
        # args.use_fp16
        # args.max_batch_size
    # Try increasing or decreasing img size (currently 768x768)
    # on file /home/frubin/Projects/Kaggle/OCR/PaddleOCR/ppocr/data/imaug/operators.py, i have hardcoded to make all images have same size
        # - instead of this, I should make a way of batching that puts images of similar size together
    # use parallelization on pre and postprocessing (dataloader??)
    images_path = os.path.join(DATA_FOLDER, images_path)
    dataframe_path = os.path.join(DATA_FOLDER, dataframe_path)
    df = pd.read_csv(dataframe_path)

    text_detector = TextE2E(args)
    all_words, all_mean_scores, all_min_scores, all_images = [], [], [], []

    starts = np.arange(0, len(df), batch_size)
    ends = starts + batch_size
    total_time = 0
    for start, end in tqdm(zip(starts, ends), total=len(starts)):
        image_ids = df.iloc[start:end]['image']
        images = [cv2.imread(os.path.join(images_path, image_id)) for image_id in image_ids]
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

        points, strs, mean_scores, min_scores, elapse = text_detector(images)
        total_time += elapse
        for i in range(0, len(strs)):
            all_words.extend(strs[i])
            all_mean_scores.extend(mean_scores[i])
            all_min_scores.extend(min_scores[i])
            all_images.extend([image_ids.iloc[i]]*len(strs[i]))

    print(total_time)
    df_res = pd.DataFrame([all_words, all_mean_scores, all_min_scores, all_images]).T
    df_res.columns = ['word', 'score_mean', 'score_min', 'image']
    return df_res


if __name__ == "__main__":
    args = utility.parse_args()
    predict_shopee_ds(args)