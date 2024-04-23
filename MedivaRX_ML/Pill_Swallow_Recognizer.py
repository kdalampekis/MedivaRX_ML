import numpy as np
import cv2
import tensorflow as tf

# Load the trained model
loaded_model = tf.saved_model.load(r"C:\Users\kostas bekis\PycharmProjects\live_face_recognition\live_face_recognition\face.py\results")

# Define the expected input shape and data type
input_shape = (1, 224, 224, 3)
input_dtype = tf.float32


# Define the function for making predictions on the loaded model
@tf.function(input_signature=[{'image': tf.TensorSpec(shape=(None, None, 224, 224, 3), dtype=tf.float32)}])
def predict_fn(inputs):
    logits = loaded_model(inputs)
    return {'output': tf.argmax(logits, axis=1, output_type=tf.int32)}


# Open a video capture object for live video stream (use appropriate video source)
video_capture = cv2.VideoCapture(0)

# Initialize a list of frames
frames = []

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Show the frame
    cv2.imshow('Video', frame)

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (224, 224))

    # Add the frame to a list
    frames.append(resized_frame)

    # If list is the disired length, proceed to prediction
    if len(frames) == 20:

        # Convert the list of frames to the right shape and type
        frames = np.array(frames)
        input_tensor = np.expand_dims(frames, axis=0)
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

        # Make a prediction
        prediction = predict_fn({'image': input_tensor})['output'].numpy()[0]

        # Display the prediction on the frame
        if prediction == 0:
            text = 'Not taking pill'
        else:
            text = 'Taking pill'

        print(text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (50, 50), font, 1, (0, 255, 0), 2)

        # Re-initialize the list of frames
        frames = []

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
