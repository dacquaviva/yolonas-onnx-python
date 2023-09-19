import cv2
import numpy as np
import onnxruntime

def preprocess_image(image, target_size=(640, 640)):
    image = cv2.resize(image, target_size)
    image_bchw = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2))
    return image, image_bchw

def show_predictions_live(image, predictions):
    num_predictions, pred_boxes, pred_scores, pred_classes = predictions

    assert num_predictions.shape[0] == 1, "Only batch size of 1 is supported by this function"

    num_predictions = int(num_predictions.item())
    pred_boxes = pred_boxes[0, :num_predictions]
    pred_scores = pred_scores[0, :num_predictions]
    pred_classes = pred_classes[0, :num_predictions]
    class_names = "#" # list of class names

    for (x1, y1, x2, y2, class_score, class_index) in zip(pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3], pred_scores, pred_classes):
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        class_name = class_names[class_index]
        label = f"{class_name}: {class_score:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def main():
    # Load ONNX model and initialize webcam
    session = onnxruntime.InferenceSession("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inputs = [o.name for o in session.get_inputs()]

    cap = cv2.VideoCapture(0)  # Initialize webcam. Adjust the parameter if necessary.

    # Create a window and set it to fullscreen
    cv2.namedWindow('Webcam Inference', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Webcam Inference', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()  # Read a frame from webcam

        if not ret:
            break

        # Preprocess the frame
        processed_frame, image_bchw = preprocess_image(frame)

        # Model inference
        result = session.run(None, {inputs[0]: image_bchw})

        # Display the predictions on the frame
        output_frame = show_predictions_live(processed_frame, result)
        
        cv2.imshow('Webcam Inference', output_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()





