import cv2
from mrcnn_handle import MaskRCNNInferenceHandler
import numpy as np
from Mask_RCNN.samples.coco import coco
from style_transfer.models import get_evaluate_model
from style_transfer.utils import process_image, get_padding, deprocess_image, remove_padding

effect = "gray"  # gray, pixelate, blur, style_transfer
original_width, original_height = 960, 540
resized_width, resized_height = 960, 540


class StyleTransferHandler():
    def __init__(self, width, height, weights_read_path):
        if height % 8 != 0:
            pad_height = (height // 8 + 1) * 8
        else:
            pad_height = height

        if width % 8 != 0:
            pad_width = (width // 8 + 1) * 8
        else:
            pad_width = width

        self.eval_model = get_evaluate_model(pad_width, pad_height)
        self.eval_model.load_weights(weights_read_path)

    def predict(self, image):
        content = process_image("", -1, -1, resize=False, image=image)
        ori_height = content.shape[1]
        ori_width = content.shape[2]

        content = get_padding(content)
        height = content.shape[1]
        width = content.shape[2]

        res = self.eval_model.predict([content])
        output = deprocess_image(res[0], width, height)
        output = remove_padding(output, ori_height, ori_width)

        return output


def pixelate_rgb(img, window):
    n, m, _ = img.shape
    n, m = n - n % window, m - m % window
    img1 = np.zeros((n, m, 3))
    for x in range(0, n, window):
        for y in range(0, m, window):
            img1[x:x + window, y:y + window] = img[x:x + window, y:y + window].mean(axis=(0, 1))
    return img1.astype(np.uint8)


if __name__ == "__main__":
    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 80

    config = {
        "model_dir": "./rcnn_model",
        "coco_model_path": "./rcnn_model/mask_rcnn_coco.h5",
        "inference_config": InferenceConfig()
    }

    # Load Mask_RCNN Model
    rcnn_model = MaskRCNNInferenceHandler(config=config)

    # Load input video stream and output video stream
    cap = cv2.VideoCapture('./resource/stab.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./resource/stab_proc_3.mp4', fourcc, 30.0, (original_width, original_height))

    if effect == "style_transfer":
        style_transfer_handler = StyleTransferHandler(resized_width, resized_height,
                                                      "./style_transfer/trained_nets/udnie_weights.h5")

    frame_window = []
    frame_window_length = 1
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = rcnn_model.inference_image([frame])
        frame_window.append((frame, results[0]))

        if len(frame_window) == frame_window_length:
            frame, results = frame_window[frame_window_length // 2]
            cv2.imshow('Frame', frame)

            front_frame, back_frame = frame, frame
            mix_rule = ""
            if effect == "gray":
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                back_frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
            elif effect == "pixelate":
                front_frame = pixelate_rgb(frame, 15)
            elif effect == "blur":
                front_frame = cv2.blur(frame, (15, 15))
            elif effect == "style_transfer":
                resized_image = cv2.resize(frame, (resized_width, resized_height))
                back_frame = style_transfer_handler.predict(resized_image)
                back_frame = cv2.resize(back_frame, (original_width, original_height))

            result_frame = rcnn_model.mix_images_stabilized(front_frame, back_frame,
                                                            [pair[1] for pair in frame_window], "person",
                                                            mix_rule="")

            cv2.imshow('Marked', result_frame)
            out.write(result_frame)

            frame_window.pop(0)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
