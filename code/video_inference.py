import cv2
from mrcnn_handle import MaskRCNNInferenceHandler
from Mask_RCNN.samples.coco import coco

if __name__ == "__main__":
    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 1

    config = {
        "model_dir": "./rcnn_model",
        "coco_model_path": "./rcnn_model/mask_rcnn_coco_person.h5",
        "inference_config": InferenceConfig()
    }

    # Load Mask_RCNN Model
    rcnn_model = MaskRCNNInferenceHandler(config=config)

    # Load input video stream and output video stream
    cap = cv2.VideoCapture('./resource/sample_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./resource/sample_output.mp4', fourcc, 30.0, (960, 540))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        cv2.imshow('Frame', frame)

        results = rcnn_model.inference_image([frame])
        result_frame = rcnn_model.visualize_result(frame, results[0])

        cv2.imshow('Marked', result_frame)
        out.write(result_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
