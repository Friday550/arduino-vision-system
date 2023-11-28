import cv2
import serial
from cvzone.SerialModule import SerialObject
import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
# from PIL import Image

def plot_to_image(image):
    """
    Shows an image of either tensor or numpy type. For troubleshooting.
    Args:
        image (tensor/numpy array): _description_
    """
    if "Tensor" in str(type(image)):
        image = tf.squeeze(image).numpy()
    if "float" in str(image.dtype) and np.max(image) > 1.:
        image = image.astype(np.int32)
    plt.imshow(image)
    plt.show()
    

# Load TFLite model
tfLiteModelPath = f'{Path.home()}\\OneDrive - Vertiv Co\\Documents\\Projects\\StickerAIChecker\\ArduinoImplementation\\arduino-vision-system\\bestTFLite.tflite'
interpreter = tf.lite.Interpreter(model_path=tfLiteModelPath)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
# * Set required input parameters as constants
TARGET_DTYPE = input_details['dtype']
TARGET_HEIGHT = input_details['shape'][1]
TARGET_WIDTH = input_details['shape'][2]
TARGET_SHAPE = input_details['shape']
INTERPOLATION_METHOD = cv2.INTER_CUBIC # Fastest among the methods that scored the highest accuracy (see resizingMethodTesting.csv in methodtesting folder)

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

foundPorts = get_ports()
connectPort = findArduino(foundPorts)

if connectPort != 'None':
    ser = serial.Serial(connectPort,baudrate = 9600, timeout = 1)  # Adjust port and baud rate accordingly
    print('Connected to ' + connectPort)
else:
    print('Connection issue')
# If port is not found, manually choose port
ser = SerialObject("COM4",baudRate=9600)
for i in foundPorts:
    print(i)

# Open a connection to the webcam (assuming it's the first camera)
cap = cv2.VideoCapture(0)

while True:
    # * Read a frame from the webcam
    ret, frame = cap.read()
    # Display the processed frame
    cv2.imshow('Processed Frame', frame)
    
    # * alter datatype, shape, and color scheme to match required specs for model
    frameResize = cv2.resize(
        frame, 
        dsize=(TARGET_HEIGHT, TARGET_WIDTH),
        interpolation=INTERPOLATION_METHOD,
        )
    # Convert frame from cv2 BGR to RGB
    frameRGB = cv2.cvtColor(frameResize, cv2.COLOR_BGR2RGB)
    # Cast to datatype required by model
    frameDType = tf.cast(frameRGB, dtype=TARGET_DTYPE)
    # expand to shape required by model
    frameExpanded = tf.reshape(frameDType, shape=TARGET_SHAPE)
    
    # * Load TFLite model and run image through
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tfLiteModelPath)
    # Clear output Tensor each iteration
    interpreter.allocate_tensors()
    # Set the input tensor
    interpreter.set_tensor(input_details['index'], frameExpanded)
    # Run inference
    interpreter.invoke()
    # Get the output tensor
    output_tensor = interpreter.tensor(output_details['index'])
    # Get the output results
    pred_prob = output_tensor() # prediction probability
    pred_result = int(tf.round(pred_prob).numpy()[0][0]) # rounded prediciton result
    
    # * Process output results
    if pred_result == 1.:  # Adjust threshold as needed
        ser.sendData([1])
        cv2.waitKey(200)
    else:
        ser.sendData([0])
        cv2.waitKey(200)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

# ampy --port / 'COM4' / C:\Users\zach.vickery\.conda\envs\pythonProject4\serial.py
