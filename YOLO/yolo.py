import cv2
import numpy as np 

net = cv2.dnn.readNet("YOLO/yolov3.weights", "YOLO/yolov3.cfg")
classes = []
with open("YOLO/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layers_names = net.getLayerNames()
output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

def box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	centers = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.5:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				centers.append([center_x, center_y])
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return centers, boxes, confs, class_ids

def xy_personCoordinate(frame):
	x_cord = 0
	y_cord = 0
	height, width, channels = frame.shape
	blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(output_layers)
	centers, boxes, confs, class_ids = box_dimensions(outputs, height, width)

	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			xx, yy = centers[i]
			label = str(classes[class_ids[i]])
			if label == "person":
				x_cord = xx
				y_cord = yy + (h/2)
	return x_cord, y_cord


def xy_ballCoordinate(frame):
	x_cord = 0
	y_cord = 0
	height, width, channels = frame.shape
	blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(output_layers)
	centers, boxes, confs, class_ids = box_dimensions(outputs, height, width)

	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			xx, yy = centers[i]
			label = str(classes[class_ids[i]])
			if label == "sports ball":
				x_cord = xx
				y_cord = yy
	return x_cord, y_cord

			
def detection(target, cnts, boxes, confs, class_ids, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			xx, yy = cnts[i]
			label = str(classes[class_ids[i]])
			if label == "person" and target == "person":
				cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 255), 1)
				cv2.putText(img, "PERSON", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1 )
				cv2.circle(img, (int(xx), int(yy + (h/2))), 3, (0, 255, 255), 4)
			elif label == "sports ball" and target == "ball":
				cv2.rectangle(img, (x - 20, y - 20), ((x + 20) + w, (y + 20) + h), (255,0,0), 2)
				cv2.putText(img, "BASKETBALL", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1 )
				cv2.circle(img, (int(xx), int(yy)), 2, (0, 0, 255), 1)


def yolo_detection(target, frame):
	height, width, channels = frame.shape
	blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(output_layers)
	centers, boxes, confs, class_ids = box_dimensions(outputs, height, width)
	detection(target, centers, boxes, confs, class_ids, frame)
