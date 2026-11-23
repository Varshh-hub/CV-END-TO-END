# REAL-TIME OBJECT DETECTION AND TRACKING

## 📌 Project Overview

This project is a real-time computer vision system that combines **YOLO11 object detection and ByteTrack object tracking** to detect, identify, monitor, and analyze multiple objects from live video streams.

Built using Python, OpenCV, and the Ultralytics framework, the system processes webcam footage, detects objects, assigns unique tracking IDs, and monitors their movement across consecutive frames.

Beyond basic object detection, the project implements additional computer vision functionalities such as object counting, movement tracking, trajectory visualization, line-crossing detection, zone monitoring, speed estimation, and real-time analytics.

The project also includes an integrated application that combines these functionalities into a single real-time object monitoring system.

---

## 🎯 Project Objectives

- Develop a real-time object detection and tracking pipeline.
- Identify and classify multiple objects using a pretrained YOLO model.
- Maintain unique tracking IDs for detected objects.
- Analyze object movement and trajectories across video frames.
- Implement object counting and movement monitoring.
- Build an integrated real-time analytics system.

---

## ⚙️ Key Features & Implementations

### 1. Object Detection

Uses the pretrained **YOLO11n (Nano)** model to identify objects in images and live video streams.

- Detects multiple objects simultaneously.
- Identifies and classifies objects into 80 supported COCO categories.
- Generates bounding boxes around detected objects.
- Displays object labels and confidence scores.
- Extracts object classes, confidence values, and bounding box coordinates.
- Generates detection summaries and class-wise object counts.

### 2. Real-Time Object Tracking

Integrates **ByteTrack** with YOLO11 to track detected objects across consecutive video frames.

- Assigns unique tracking IDs to detected objects.
- Maintains object identities across frames.
- Tracks multiple objects simultaneously.
- Identifies active objects in the current frame.
- Maintains tracking history for previously detected objects.

### 3. Object ID Management

Implements tracking history management to maintain information about individual objects.

- Stores unique tracking IDs and corresponding object classes.
- Records the first and last observed frames.
- Maintains the number of frames each object has been tracked.
- Stores recent object positions using a limited-length history.
- Generates structured tracking reports using Pandas.

### 4. Real-Time Object Counting

Implements object counting using detection results and tracking IDs.

- Counts currently visible objects.
- Maintains the total number of unique tracking IDs observed.
- Calculates class-wise object counts.
- Displays object statistics directly on the live video feed.
- Generates object counting summaries.

### 5. Line Crossing Detection

Implements virtual line monitoring to identify objects moving across a predefined boundary.

- Creates a virtual line within the video frame.
- Detects objects moving from left to right.
- Detects objects moving from right to left.
- Uses tracking IDs to associate movement with individual objects.
- Maintains separate directional crossing counts.
- Displays total crossing statistics in real time.

### 6. Zone Monitoring

Implements a predefined monitoring region to analyze object entry and exit movements.

- Defines a rectangular monitoring zone within the video frame.
- Identifies objects entering and leaving the designated region.
- Maintains individual object zone-status information.
- Calculates total zone entries and exits.
- Displays current zone occupancy.
- Provides real-time zone monitoring statistics.

### 7. Object Trajectory Visualization

Visualizes the movement paths of tracked objects over consecutive frames.

- Records the center coordinates of detected objects.
- Maintains recent positional history for each tracking ID.
- Draws movement trails across video frames.
- Displays object center points and tracking identifiers.
- Helps visualize the movement patterns of individual objects.

### 8. Object Speed Estimation

Implements approximate movement speed estimation using changes in object positions over time.

- Calculates displacement using consecutive object center coordinates.
- Measures elapsed time between observations.
- Estimates movement speed in pixels per second (px/s).
- Classifies objects as moving or stationary using a configurable threshold.
- Displays estimated speed alongside tracking information.

*Note: Speed is estimated in image pixels per second and is not calibrated to real-world physical speed.*

### 9. Real-Time Analytics Dashboard

Develops a live analytics dashboard that combines multiple object monitoring metrics.

The dashboard displays:

- Currently visible objects.
- Total unique tracking IDs.
- Class-wise object counts.
- Average estimated object speed.
- Current FPS and average FPS.
- Left-to-right crossing count.
- Right-to-left crossing count.
- Current monitoring-zone occupancy.
- Total zone entries and exits.

### 10. Integrated Object Detection & Tracking System

Combines the individual functionalities into a single real-time computer vision application.

The integrated system performs:

- YOLO11-based object detection.
- ByteTrack-based object tracking.
- Unique object ID management.
- Object counting and classification.
- Line-crossing detection.
- Zone entry and exit monitoring.
- Object speed estimation.
- Trajectory visualization.
- Real-time analytics and dashboard display.

This provides a unified interface for monitoring object detection, tracking, and movement-related statistics through a live webcam feed.

---

## 🔄 System Workflow

1. **Input Acquisition:** Captures live video frames using OpenCV.
2. **Object Detection:** Processes frames using the pretrained YOLO11n model.
3. **Object Classification:** Identifies detected objects and extracts their class labels, confidence scores, and bounding boxes.
4. **Object Tracking:** Uses ByteTrack to associate detections across consecutive frames.
5. **ID Management:** Assigns and maintains unique tracking IDs.
6. **Movement Analysis:** Calculates object positions, trajectories, and approximate movement speeds.
7. **Spatial Monitoring:** Detects virtual line crossings and zone entry/exit events.
8. **Analytics:** Calculates object counts, movement statistics, and FPS.
9. **Visualization:** Displays annotated video frames and real-time analytics.

---

## 🛠️ Technologies & Libraries

| Technology | Application |
|---|---|
| Python | Core programming language |
| YOLO11n | Pretrained object detection model |
| ByteTrack | Multi-object tracking |
| OpenCV | Image processing and video capture |
| PyTorch | Deep learning framework |
| Ultralytics | YOLO model integration |
| NumPy | Numerical and coordinate operations |
| Pandas | Detection reports and tracking summaries |
| Matplotlib | Data visualization |
| Seaborn | Visualization styling |

---

## 💻 Hardware Acceleration

The project automatically selects an available computation device:

- CUDA – NVIDIA GPU acceleration.
- MPS – Apple Silicon GPU acceleration.
- CPU – Standard processor execution.

The notebook was executed using **Apple Silicon MPS acceleration**.

---

## 📊 Sample Execution Results

The notebook includes the following demonstration outputs:

| Experiment | Observed Result |
|---|---|
| Image object detection | 5 objects detected |
| Average detection confidence | 83.75% |
| Object categories detected | Person, Bus |
| Integrated system | 29.42 average FPS |
| Line crossing test | 1 left-to-right and 1 right-to-left crossing |
| Zone monitoring test | 2 entries recorded |
| Trajectory visualization | Object movement history recorded |

*These are sample outputs from individual notebook executions, not benchmark results or guaranteed performance.*

---

## 🌍 Real-World Applications

- Intelligent video surveillance.
- Traffic and vehicle monitoring.
- Pedestrian movement analysis.
- Crowd monitoring and occupancy tracking.
- Restricted-area monitoring.
- Retail customer movement analytics.
- Industrial object monitoring.
- Smart security and automation systems.

---

## 📚 Key Learning Outcomes

- Implementing pretrained YOLO models for object detection.
- Integrating ByteTrack for multi-object tracking.
- Understanding object association and tracking ID management.
- Processing live video streams using OpenCV.
- Implementing object counting and trajectory analysis.
- Developing line-crossing and zone-monitoring algorithms.
- Estimating object movement using positional data.
- Building integrated real-time computer vision applications.

---

## 🚀 Future Enhancements

- Video file input and processed video export.
- Persistent tracking reports and historical analytics.
- Interactive monitoring-zone configuration.
- Object-specific filtering and monitoring.
- Web-based real-time analytics dashboard.

---

**Project Category:** Computer Vision | Deep Learning | Object Detection | Multi-Object Tracking | Video Analytics

**Developed using:** Python, YOLO11, ByteTrack, OpenCV, and PyTorch.

**Author:** Varsha A.
