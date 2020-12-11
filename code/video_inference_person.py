import cv2
from mrcnn_handle import MaskRCNNInferenceHandler
import skimage.io
import numpy as np

if __name__ == "__main__":
    # Load Mask_RCNN Model
    rcnn_model = MaskRCNNInferenceHandler()

    # Load input video stream and output video stream
    cap = cv2.VideoCapture('./resource/sample_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./resource/sample_output_person.mp4', fourcc, 30.0, (960, 540))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        cv2.imshow('Frame', frame)

        results = rcnn_model.inference_image([frame])
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)

        result_frame = rcnn_model.mix_images(frame, frame_gray, results[0], "person")

        cv2.imshow('Marked', result_frame)
        out.write(result_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
