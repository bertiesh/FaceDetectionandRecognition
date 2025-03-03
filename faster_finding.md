Tested using ArcFace + yolov8 with threshold 0.48 on 500 images
benchmark

using img preprocessing:
The time is reduced to 540s but the accuracy is also reduced to 0.69

using multiprocessing(CPU):
This is not as good as using ThreadPoolExecutor(660s), but it is a simple way to determine number of workers

using multithreading:
MAX_WORKERS = 4  652s
MAX_WORKERS = 5  638s
MAX_WORKERS = 6  642s
