from imageai.Detection import ObjectDetection
import os

exec_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(exec_path, "C:/Users/Hladkyi Dmytro/Desktop/IT Oprogromowanie/Python_Projects/project3_Object recognition/yolov3.pt"))
detector.loadModel()

custom_objects = detector.CustomObjects(umbrella=True)

detections, objects_path = detector.detectObjectsFromImage(custom_objects=custom_objects, 
    input_image=os.path.join(
    exec_path, "C:/Users/Hladkyi Dmytro/Desktop/IT Oprogromowanie/Python_Projects/project3_Object recognition/image.jpg"), 
    output_image_path=os.path.join(exec_path, "C:/Users/Hladkyi Dmytro/Desktop/IT Oprogromowanie/Python_Projects/project3_Object recognition/imageout1.jpg"), minimum_percentage_probability=50,  
    extract_detected_objects=True)



for eachObject, eachObjectPath in zip(detections, objects_path):
    print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("Object's image saved in " + eachObjectPath)
    print("--------------------------------")


# from imageai.Detection import VideoObjectDetection
# import os

# execution_path = os.getcwd()

# detector = VideoObjectDetection()
# detector.setModelTypeAsYOLOv3()
# detector.setModelPath( os.path.join(execution_path , "C:/Users/Hladkyi Dmytro/Desktop/IT Oprogromowanie/Python_Projects/project3_Object recognition/yolov3.pt"))
# detector.loadModel()

# video_path = detector.detectObjectsFromVideo(
#     input_file_path=os.path.join(execution_path, "C:/Users/Hladkyi Dmytro/Desktop/IT Oprogromowanie/Python_Projects/project3_Object recognition/wideo.mp4"),
#     output_file_path=os.path.join(execution_path, "C:/Users/Hladkyi Dmytro/Desktop/IT Oprogromowanie/Python_Projects/project3_Object recognition/traffic_detected"), 
#     frames_per_second=20, 
#     log_progress=True)
# print(video_path)



# https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Detection ---->download RetinaNet



# pip install cython pillow>=7.0.0 numpy>=1.18.1 opencv-python>=4.1.2 torch>=1.9.0 --extra-index-url https://download.pytorch.org/whl/cpu torchvision>=0.10.0 --extra-index-url https://download.pytorch.org/whl/cpu pytest==7.1.3 tqdm==4.64.1 scipy>=1.7.3 matplotlib>=3.4.3 mock==4.0.3

# pip install cython pillow>=7.0.0 numpy>=1.18.1 opencv-python>=4.1.2 torch>=1.9.0 --extra-index-url https://download.pytorch.org/whl/cu102 torchvision>=0.10.0 --extra-index-url https://download.pytorch.org/whl/cu102 pytest==7.1.3 tqdm==4.64.1 scipy>=1.7.3 matplotlib>=3.4.3 mock==4.0.3

# pip install pycocotools@git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI

# pip install imageai --upgrade


# https://imageai.readthedocs.io/en/latest/video/index.html ----> https://imageai.readthedocs.io/en/latest/video/index.html