import streamlit as st
import numpy as np



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


def main():
    st.title("Sekah")
    st.sidebar.title ("Settings")
    st.sidebar.subheader ("Parameters")
    st.markdown (
    """
    <style>
    [data-testid="stSidebar"][arta-expanded-"true"] > div:first-child {
    width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded-"true"] > div:first-child {
    width: 300px;
    margin-left: -300px;
    }
    </style>
    """,
    unsafe_allow_html=True

 )

app_mode = st.sidebar.selectbox('The pages', ['About Sekah', 'Try the model'])

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
