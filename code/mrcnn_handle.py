from Mask_RCNN.samples.coco import coco
from Mask_RCNN.mrcnn import model as model_lib
from Mask_RCNN.mrcnn import utils
from Mask_RCNN.mrcnn import visualize
import os
import numpy as np
import io
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from PIL import Image, ImageFilter


class MaskRCNNInferenceHandler:
    def __init__(self, config=None):
        if not config:
            config = self.default_config()
        self.config = config
        self.coco_class_names = self.get_coco_class_names()
        self.color_map = self.assign_color_by_class_names(self.coco_class_names)

        if not os.path.exists(self.config["coco_model_path"]):
            utils.download_trained_weights(self.config["coco_model_path"])

        self.model = model_lib.MaskRCNN(mode="inference", model_dir=self.config["model_dir"],
                                        config=self.config["inference_config"])
        self.model.load_weights(self.config["coco_model_path"], by_name=True)

    def default_config(self):
        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = {
            "model_dir": "./rcnn_model",
            "coco_model_path": "./rcnn_model/mask_rcnn_coco.h5",
            "inference_config": InferenceConfig()
        }

        return config

    def inference_image(self, images):
        if not isinstance(images, list) and isinstance(images, np.ndarray):
            images = [images]

        results = self.model.detect(images, verbose=0)

        return results

    def get_coco_class_names(self):
        class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']

        return class_names

    def assign_color_by_class_names(self, class_names):
        colors = visualize.random_colors(len(class_names))

        return colors

    def visualize_result(self, image, r):
        def rgba2rgb(rgba, background=(255, 255, 255)):
            row, col, ch = rgba.shape

            if ch == 3:
                return rgba

            assert ch == 4, 'RGBA image has 4 channels.'

            rgb = np.zeros((row, col, 3), dtype='float32')
            r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

            a = np.asarray(a, dtype='float32') / 255.0

            R, G, B = background

            rgb[:, :, 0] = r * a + (1.0 - a) * R
            rgb[:, :, 1] = g * a + (1.0 - a) * G
            rgb[:, :, 2] = b * a + (1.0 - a) * B

            return np.asarray(rgb, dtype='uint8')

        colors_assigned = [self.color_map[i] for i in r["class_ids"]]

        image_size = image.shape
        fig = plt.figure(figsize=(image_size[1]/100.0, image_size[0]/100.0), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        visualize.display_instances(image, r["rois"], r["masks"], r["class_ids"], self.coco_class_names,
                                    r["scores"], colors=colors_assigned, ax=ax)

        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw')
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()
        plt.close(fig)

        return rgba2rgb(img_arr)

    def mix_images_stabilized(self, image_ref, image_ret, rs, class_name, mix_rule=""):
        target_class_index = self.coco_class_names.index(class_name)

        final_masks = []
        for r in rs:
            final_mask = np.zeros([image_ref.shape[0], image_ref.shape[1]]).astype(np.bool)
            for i in range(len(r["class_ids"])):
                if r["class_ids"][i] != target_class_index:
                    continue

                final_mask = np.logical_or(r["masks"][:, :, i], final_mask)

            final_masks.append(final_mask)

        final_mask = np.count_nonzero(np.stack(final_masks), axis=0) >= (len(final_masks) + 1 // 2)

        if mix_rule == "smooth":
            mask_im = Image.fromarray(final_mask.astype(np.uint8)*255)
            mask_blur = mask_im.filter(ImageFilter.GaussianBlur(5))

            im = Image.composite(Image.fromarray(image_ref), Image.fromarray(image_ret), mask_blur)
            image_ret = np.asarray(im)
        else:
            for i in range(image_ref.shape[2]):
                image_ret[:, :, i] = image_ret[:, :, i] * np.logical_not(final_mask) + image_ref[:, :, i] * final_mask

        return image_ret

    def mix_images(self, image_ref, image_ret, r, class_name):
        target_class_index = self.coco_class_names.index(class_name)

        final_mask = np.zeros([image_ref.shape[0], image_ref.shape[1]]).astype(np.bool)
        for i in range(len(r["class_ids"])):
            if r["class_ids"][i] != target_class_index:
                continue

            final_mask = np.logical_or(r["masks"][:, :, i], final_mask)

        for i in range(image_ref.shape[2]):
            image_ret[:, :, i] = image_ret[:, :, i] * np.logical_not(final_mask) + image_ref[:, :, i] * final_mask

        return image_ret

