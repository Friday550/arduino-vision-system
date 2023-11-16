import cv2
import serial
import serial.tools.list_ports
import tensorflow as tf
# import ampy
# import numpy as np
# from cvzone.SerialModule import SerialObject
# from oauthlib.uri_validate import port

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='converted_model.tflite')
interpreter.allocate_tensors()

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

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open a connection to the webcam (assuming it's the first camera)
cap = cv2.VideoCapture(0)

while True:
    # Clear output Tensor each iteration
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path='converted_model.tflite')
    interpreter.allocate_tensors()

    # Read a frame from the webcam
    ret, frame = cap.read()
    # print(ret)
    tensor_frame = tf.image.resize_with_pad(
        tf.reshape(frame, shape=(1, 480, 640, 3)),
        224,
        224
    )

    print(tensor_frame.shape)
    # input()
    tensor_frame = tf.constant(tensor_frame)
    # print(type(frame))

    # Normalize the pixel values to be between 0 and 1
    # input_data = frame / 255.0
    # input_data = tf.cast(input_data, tf.float32)
    # Expand dimensions to create a batch-size of 1
    # input_data = tf.expand_dims(input_data, axis=0)
    # print('INPUT: ', input_data)
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], tensor_frame)

    # Run inference
    interpreter.invoke()

    # Display the processed frame
    cv2.imshow('Processed Frame', frame)

    # Get the output tensor
    output_tensor = interpreter.tensor(interpreter.get_output_details()[0]['index'])

    # Get the output results
    output_results = output_tensor()
    print("Output Results:", output_results)  # Add this line to print the output_results
    list(map(int, output_results[0]))

    # Process output results
    if output_results[0][0] > 0.5:  # Adjust threshold as needed
        print("Model predicts a positive outcome!")
        ser.write([1, 0])
        cv2.waitKey(500)
    else:
        print("Model predicts a negative outcome.")
        ser.write([0, 1])
        cv2.waitKey(500)

    print("finished a run")
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

# ampy --port / 'COM4' / C:\Users\zach.vickery\.conda\envs\pythonProject4\serial.py