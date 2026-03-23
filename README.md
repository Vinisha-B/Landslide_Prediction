🌍 Landslide Prediction Using Satellite Images

🔹 Project Overview

This project uses Machine Learning and Deep Learning to detect landslide areas from satellite images. Users can upload a satellite image to a web app built with Streamlit, and the app predicts whether the image shows a landslide.

Dataset: Satellite images of landslide and non-landslide regions
Model: Pretrained MobileNetV2 with transfer learning
Deployment: Interactive Streamlit app
🔹 Features
Upload satellite images (jpg/png)
Real-time prediction of landslide areas
Easy-to-use web interface
Uses a trained deep learning model for high accuracy
🔹 Folder Structure
landslide-project/
├── archive/                  # Original dataset (optional)
├── dataset/                  # Train/Test split for ML
├── model/
│   └── landslide_model.h5    # Trained model
├── app.py                    # Streamlit app
├── train.py                  # Model training script
├── split_dataset.py          # Dataset split script
├── requirements.txt          # Dependencies
└── README.md
🔹 Installation

Clone this repository:

git clone https://github.com/Vinisha-B/Landslide_Prediction.git
cd Landslide_Prediction

Install dependencies:

pip install -r requirements.txt
🔹 Run Locally

To launch the Streamlit app locally:

python -m streamlit run app.py

Upload a satellite image and see the prediction!

🔹 Model Training

If you want to retrain the model:

Make sure the dataset is in dataset/train and dataset/test
Run:
python train.py

The model will be saved to model/landslide_model.h5.

🔹 Deployment

The app can be deployed to Streamlit Cloud:

Go to https://streamlit.io/cloud
Connect your GitHub repo
Deploy app.py
The live app will be available online for anyone

Example live app link: [Add your Streamlit Cloud URL here]

🔹 Technologies Used
Python 3.x
TensorFlow / Keras
Streamlit
NumPy & Pillow
Transfer Learning (MobileNetV2)
🔹 Screenshots

Upload Page:


Prediction Result:


Replace placeholders with your actual screenshots.

🔹 Author

Vinisha Baskar 💚

GitHub: Vinisha-B
Email: your-email@example.com
