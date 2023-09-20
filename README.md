<img src="https://github.com/NouraAlqassim/Sekah/blob/main/images/logo.png" alt="drawing" width="250"/>
Visual pollution detection model that detects defective sidewalk & speed bumps, pothole, garbage, and Jersey barrier gaps to prevent animals and children from falling in construction ereas.

# Objectives
- Make Roads maintenance process faster and more efficient.
- Reduce environmental impact.
- Enhance public safety.
- Enhance urban quality of life
- Improve aesthetics.

# Dataset
The data set was obtained from Roboflow

**The link:** https://universe.roboflow.com/visual-pollution-056la/visual-pollution-pc2as/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true

- We performed data cleaning to remove duplicate images in both training and testing sets.
- Also we made modifications to some of the classes to make it more clear.

# The model
We fine-tuned YOLO-NAS pretrained model and modified some of its augmentation techniques to make it more suitable for out project requirement.

