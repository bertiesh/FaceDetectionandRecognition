#### Image Preprocessing
Reduce size by 50%

#### Multiprocessing
**Process Pool**: Use ProcessPoolExecutor instead of ThreadPoolExecutor to create a pool of worker processes.
**Model Instances**: Add a model cache to avoid recreating the model for each task.
**Dynamic Workers**: The number of workers is calculated based on your CPU count, leaving one core free for system tasks.
**Optimization for Small Batches**: For 1-2 images, the code skips multiprocessing to avoid the overhead of creating processes.
**Partial Functions**: Use functools.partial to create a fixed-parameter version of the processing function, making it cleaner to pass to the executor.

#### Multithread
**Concurrent Processing**: Add a ThreadPoolExecutor from the concurrent.futures module to process multiple images in parallel.
**Thread Safety**: Add a Lock for database operations.
**Worker Function**: Creat a separate process_face_match function to handle individual image processing.
**Configurable Workers**: Add a MAX_WORKERS constant (set to 5) 
**Error Handling**: Improve error handling to catch and report exceptions from worker threads.

#### Load Balancing
**Dynamic Worker Pool**
Adjust the number of worker threads based on CPU usage, memory availability, and GPU utilization
Use configurable min/max worker limits (default 2-8)
Include periodic system monitoring to make smart scaling decisions

**System Health Monitoring**
Implement circuit breaker pattern to prevent system overload
Automatically reject requests when the system is under heavy load
Include recovery mechanism after overload conditions subside

**Task Prioritization**
Sort images by complexity before processing
Process simpler images first for better user experience
Use file size as a proxy for complexity (can be extended)

**Performance Metrics and Logging**
Track and log processing time for individual images and batches
Provide detailed diagnostics about system conditions
Record worker allocation decisions for troubleshooting


#### Tests
Tested using ArcFace + yolov8 with threshold 0.48 on 500 images
benchmark
700+s

using img preprocessing:
The time is reduced to 540s but the accuracy is also reduced to 0.69

using multiprocessing(CPU):
This is not as good as using ThreadPoolExecutor(660s), but it is a simple way to determine number of workers

using multithreading:
MAX_WORKERS = 4  652s
MAX_WORKERS = 5  638s
MAX_WORKERS = 6  642s
