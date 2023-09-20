import streamlit as st
import numpy as np
import io
from PIL import Image
import pathlib
import cv2
import pathlib
from io import BytesIO

from super_gradients.training import Trainer as super_gradients_Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
)
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095
)
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback



def load_image():
    uploaded_file = st.file_uploader(label=' ')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data,caption='Original Image', use_column_width=True)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_model():
    mymodel = models.get(
    model_name='yolo_nas_s',
    checkpoint_path='ckpt_best.pth',
    num_classes=5
    )
    return mymodel


def predict(predictor, image):
    
    # Perform prediction
    image.thumbnail((640, 640))

    prediction = predictor.predict(image)

    # Check the number of bounding boxes
    num_boxes = len(prediction.labels)
    print("Number of bounding boxes:", num_boxes)

    # Convert PIL image to OpenCV format (BGR)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Iterate over the bounding boxes and draw them on the image
    for bbox in prediction.labels:
        x, y, width, height = bbox.x, bbox.y, bbox.width, bbox.height
        class_name = bbox.classifier
        
        # Draw rectangle
        cv2.rectangle(image_cv, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        # Add class name as text on the bounding box
        cv2.putText(image_cv, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image with bounding boxes using streamlit
    st.image(image_cv, channels="BGR",caption='Predicted Image', use_column_width=True)

def main():
    # Streamlit app code
    st.markdown("<h2 style='text-align: center;'>Model Demo</h2>", unsafe_allow_html=True)

    model = load_model()

    # Create two square containers using 'beta_columns' layout
    col1, col2 = st.columns(2)
    
    # Container 1: Display the original image
    with col1:
        st.markdown("<h3 style='text-align: center;'>Original Image</h3>", unsafe_allow_html=True)
        image = load_image()

    # Container 2: Display the predicted image
    with col2:
        st.markdown("<h3 style='text-align: center;'>Predicted Image</h3>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        result = st.button('Perform Prediction')
        
        if result:
            st.markdown("<br>", unsafe_allow_html=True)
            st.write('Calculating results...')
            prediction = predict(model, image)
if __name__ == '__main__':
    main()


