# https://github.com/JustinGuese/mtcnn-face-extraction-eyes-mouth-nose-and-speeding-it-up/blob/master/MTCNN%20example.ipynb
#   Advanced version


from facenet_pytorch import MTCNN
from PIL import Image
import torch
from imutils.video import FileVideoStream
import cv2
import time
import glob
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

class FastMTCNN(object):
    """Fast MTCNN implementation. Modified to return boxes"""
    
    def __init__(self, stride, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.
        
        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.
        
        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        
    def __call__(self, frames):
        """Detect faces in frames using strided MTCNN."""
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                    for f in frames
            ]
                      
        boxes, probs = self.mtcnn.detect(frames[::self.stride])

        faces = []
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            for box in boxes[box_ind]:
                box = [int(b) for b in box]
                faces.append(frame[box[1]:box[3], box[0]:box[2]])
        
        return (faces, boxes)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define our extractor
fast_mtcnn = FastMTCNN(
    stride=4,
    resize=1, # 0.5 no resize will allow boxes to fit nicely
    margin=14,
    factor=0.6,
    keep_all=True,
    device=device
)

def draw_facebox(filename, result_list):
    # load the image
    data = plt.imread(filename)
    # plot the image
    plt.imshow(data)
    # get the context for drawing boxes
    ax = plt.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box'] # there is no BOX !
        # create the shape
        rect = plt.Rectangle((int(x), int(y)), int(width), int(height), fill=False, color='orange')
        # draw the box
        ax.add_patch(rect)
    # show the plot
    plt.show()

def run_detection(fast_mtcnn, filename):
    frames = []
    frames_processed = 0
    faces_detected = 0
    batch_size = 60
    start = time.time()
    v_cap = FileVideoStream(filename).start()
    v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    for j in range(v_len):
        frame = v_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if len(frames) >= batch_size or j == v_len - 1:
            faces, boxes = fast_mtcnn(frames)
            frames_processed += len(frames)
            faces_detected += len(faces)
            frames = []
            fps = frames_processed / (time.time() - start)
            print(
                f'                          Frames per second: {fps:.3f},',
                f'faces detected: {faces_detected}\r',
                end=''
            )
            print("========================")
            print(boxes[0]) 
            print(faces_detected)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR )
            for boxNo in range(faces_detected):
                xStart = int(boxes[0][boxNo, 0])
                yStart = int(boxes[0][boxNo, 1])
                xEnd = int(boxes[0][boxNo, 2])
                yEnd = int(boxes[0][boxNo, 3])            
                # [[148.05846  93.21905 217.33783 175.8501 ]]       
                print( xStart, yStart, xEnd, yEnd )
                cv2.rectangle(frame,
                    (xStart, yStart),
                    (xEnd, yEnd),
                    (255, 255, 255),
                    thickness=1)
    cv2.rectangle(frame,(0,0),(550,30),(0, 0, 0),-1)
    label = "Number of faces detected : {:} Frames per second : {:.3f}".format(faces_detected, fps)
    cv2.putText(frame, label, (20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),1)
    cv2.imshow("Result",frame)
    cv2.waitKey(0)
    v_cap.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    filename = "crowd-4.jpg"
    run_detection(fast_mtcnn, filename)
