from setuptools import setup, find_packages

setup(
    name="aerial_classification",
    version="0.1.0",
    author="Amogh Samadhiya",
    description="Aerial Object Classification (Bird vs Drone) and Detection Project",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.10.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics",
        "streamlit",
        "numpy",
        "pandas",
        "matplotlib",
        "pillow",
        "opencv-python-headless",
        "scikit-learn"
    ],
    python_requires=">=3.8",
)