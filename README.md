# Project-EndoLite
An Efficient, Quantized Classifier for Real-Time Endoscopic Anomaly Detection on Edge Devices
1.Abstract
This project presents a lightweight deep learning framework for the classification of four distinct medical conditions from endoscopic images: Bleeding, Lymphangiectasia, Polyp, and Ulcer. By leveraging a comprehensive suite of model optimization techniques—including knowledge distillation, structured pruning, and post-training quantization—the framework is engineered to deliver high classification accuracy while maintaining a minimal computational footprint. The ultimate goal is to produce an efficient and reliable model suitable for deployment on edge or mobile devices, enabling real-time automated analysis of medical imagery.


2.Four-Class Medical Image Classification:
Trained to accurately distinguish between Bleeding, Lymphangiectasia, Polyp, and Ulcer.
Teacher-Student Architecture: Utilizes a larger "teacher" model (ResNet-18) to train a smaller, more efficient "student" model (MobileNetV2) via knowledge distillation.
Structured Pruning: Implements L1-norm pruning to remove redundant channels and reduce model size without significant accuracy loss
Post-Training Quantization: Further compresses the model and accelerates inference by converting weights to lower-precision formats.
Edge-Ready TFLite Conversion: The final model is converted to the TensorFlow Lite format, making it ready for deployment on mobile or embedded systems.


3.Dataset Structure
This project uses a dataset of medical images organized by class. For the code to run correctly, your dataset should be structured as follows. Note that the image you provided shows the classes split into train and val directories, which is not the standard ImageFolder format. The code expects the train and val folders to be at the top level.
dataset/
├── train/
│   ├── Bleeding/
│   │   ├── img1.jpg
│   │   └── ...
│   ├── Lymphangiectasia/
│   ├── Polyp/
│   └── Ulcer/
└── val/
    ├── Bleeding/
    ├── Lymphangiectasia/
    ├── Polyp/
    └── Ulcer/


4.Pipeline
1. Train the Teacher Model
Train the larger ResNet-18 model which will act as the teacher
        ""python train_teacher.py""
2. Train the Student Model 
Train the lightweight MobileNetV2 student model using the knowledge distilled from the teacher
     ""python train_student.py""
3. Prune the Student Model
prune the student model to further reduce its size and complexity
     ""python model_student.py""
4. Convert the Pruned Model to TFLite
Convert the final pruned PyTorch model to the TensorFlow Lite format for deployment
     "" python convert_to_tflite.py""
5. Quantize the TFLite Model
Apply post-training quantization to the .tflite model for final optimization.
     ""python quantize_tflite.py""



