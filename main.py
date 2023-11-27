import cv2
import serial
from cvzone.SerialModule import SerialObject
import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from pathlib import Path
import io
from PIL import Image
ser = SerialObject("COM4")
# Required size to input into Efficientnet tuned model
# data type required by model. EfficientNet will normalize images within the model so pixels should be between 0 and 255
RESIZING_METHODS = [
    'bilinear',
    'lanczos3',
    'lanczos5',
    'bicubic',
    'gaussian',
    'nearest',
    'area',
    'mitchellcubic',
]
resizeMethod = RESIZING_METHODS[5]
# Load TFLite model
tfLiteModelPath = f'{Path.home()}\\OneDrive - Vertiv Co\\Documents\\Projects\\StickerAIChecker\\ArduinoImplementation\\arduino-vision-system\\bestTFLite.tflite'
interpreter = tf.lite.Interpreter(model_path=tfLiteModelPath)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
TARGET_DTYPE = input_details['dtype']
TARGET_HEIGHT = input_details['shape'][1]
TARGET_WIDTH = input_details['shape'][2]
TARGET_SHAPE = input_details['shape']
# Initialize serial connection
def get_ports():
    ports = serial.tools.list_ports.comports()
    return ports
def findArduino(portsFound):
    commPort = 'None'
    numConnection = len(portsFound)
    for i in range(0,numConnection):
        port = foundPorts[i]
        strPort = str(port)
        if 'Arduino' in strPort:
            splitPort = strPort.split(' ')
            commPort = (splitPort[0])
    return commPort
def plot_to_image(image):
    if "Tensor" in str(type(image)):
        image = tf.squeeze(image).numpy()
    if "float" in str(image.dtype) and np.max(image) > 1.:
        image = image.astype(np.int32)
    plt.imshow(image, 
            #    interpolation='nearest', 
            #    cmap='none'
               )
    plt.show()
foundPorts = get_ports()
connectPort = str(foundPorts[2]).split(' ')[0]
# findArduino(foundPorts)

# if connectPort != 'None':
#     ser = serial.Serial(connectPort,baudrate = 9600, timeout = 1)  # Adjust port and baud rate accordingly
#     print('Connected to ' + connectPort)
# else:
#     print('Connection issue')

# Open a connection to the webcam (assuming it's the first camera)
cap = cv2.VideoCapture(0)

while True:
    # Clear output Tensor each iteration
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tfLiteModelPath)
    interpreter.allocate_tensors()
    # Read a frame from the webcam
    ret, frame = cap.read()
    # * Convert frame from cv2 BGR to RGB
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resized_frame = tf.image.resize(
        frameRGB,
        size=(TARGET_HEIGHT, TARGET_WIDTH),
        method=RESIZING_METHODS[5],
    # antialias=True,
    # interpolation=cv2.INTER_CUBIC,
    )
    frameDType = tf.cast(resized_frame, dtype=TARGET_DTYPE)
    frameExpanded = tf.reshape(frameDType, shape=TARGET_SHAPE)
    # normalized_tensor = frameExpanded / 255.
    # tensor_frame = cv2.resize(
    #     img, 
    #     dsize=(TARGET_SHAPE[1], TARGET_SHAPE[2]),
    #     interpolation=resizeMethod,
    #     # antialias=True,
    #     # interpolation=cv2.INTER_CUBIC,
    #     )
    # * Display the processed frame\

    cv2.imshow('Processed Frame', frame)
    # plot_to_image(frameExpanded)

    # Set the input tensor
    interpreter.set_tensor(input_details['index'], frameExpanded)
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output_tensor = interpreter.tensor(output_details['index'])
    # Get the output results
    pred_prob = output_tensor() # prediction probability
    # print("type: ", type(pred_prob), type(output_tensor), type(output_tensor()))
    # print("Output Results:", pred_prob)  # Add this line to print the output_results
    pred_result = int(tf.round(pred_prob).numpy()[0][0]) # rounded prediciton result
    # Process output results
    print(pred_result)
    if pred_result == 1.:  # Adjust threshold as needed
        ser.sendData([1])
        cv2.waitKey(200)
    else:
        ser.sendData([0])
        cv2.waitKey(200)

    # print("finished a run")
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

# ampy --port / 'COM4' / C:\Users\zach.vickery\.conda\envs\pythonProject4\serial.py
