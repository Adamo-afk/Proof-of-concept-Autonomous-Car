# import the necessary packages
import tensorflow as tf
from config_directory import config
# import numpy as np
# import cv2

# def next_action():
#     if action == config.CLASSES[0]:
#         print(config.CLASSES[0])
#     elif action == config.CLASSES[1]:
#         print(config.CLASSES[1])
#     elif action == config.CLASSES[2]:
#         print(config.CLASSES[2])
#     elif action == config.CLASSES[3]:
#         print(config.CLASSES[3])

# load the trained model from disk
print("[INFO] loading model...")
model = tf.keras.models.load_model(config.MODEL_PATH)
model.summary()

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# with tf.io.gfile.GFile('my_model.tflite', 'wb') as f:
#     f.write(tflite_model)

model.save_weights('model_weights.h5')

# # load each frame from the camera and then clone it so we can draw on it later
# video_feed = cv2.VideoCapture(0)
# frame_counter = 0

# while True:
    
#     ret, frame = video_feed.read()
#     output = frame

#     if not ret:
#         break

#     frame_counter += 1
    
#     if frame_counter % 5 == 0:
#         # the model was trained on RGB ordered images but OpenCV represents
#         # images in BGR order, so swap the channels, and then resize the image
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         frame = cv2.resize(frame, (config.IMAGE_SIZE[-1], config.IMAGE_SIZE[-1]))

#         # # convert the image to a floating point data type and perform mean
#         # # subtraction
#         # frame = frame.astype("float32")
#         # mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")
#         # frame -= mean

#         # convert the image to a floating point data type and perform normalization
#         frame = frame.astype("float32")
#         frame = frame/255
            
#         # pass the image through the network to obtain the predictions
#         predictions = model.predict(np.expand_dims(frame, axis=0))[0]
#         # print("Ahead: {:.2f} | STOP: {:.2f} | Left: {:.2f} | Right: {:.2f}".format(predictions[0], predictions[1], predictions[2], predictions[3]))
#         i = np.argmax(predictions)

#         if frame_counter == 5:
#             action = config.CLASSES[1]

#         if predictions[i] >= 0.9:
#             print(config.CLASSES[i])
#             action = config.CLASSES[i]
#         else:
#             next_action()

#         # draw the prediction on the output image
#         text = "{}: {:.2f}%".format(config.CLASSES[i], predictions[i] * 100)
#         cv2.putText(output, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
#         # show the output image
#         cv2.imshow("Output", output)

#     # Exit the loop if 'q' is pressed
#     if cv2.waitKey(1) == ord('q'):
#         break   

# video_feed.release()

