# Criminal Recognition System

This project is a Criminal Recognition System that uses OpenCV and Python. The system uses a camera to capture images of individuals and then compares them with the images in the database to detect criminals.

## Installation

1. Clone the repository to your local machine. ``` https://github.com/Aaryan1901/Criminal_Detection_System.git
 ```
2. Install the required packages using ```pip install -r requirements.txt```.
3. Download the dlib models from https://drive.google.com/drive/folders/12It2jeNQOxwStBxtagL1vvIJokoz-DL4?usp=sharing and place the data folder inside the repo

## Usage

1. Collect the Faces Dataset by running ``` python get_faces_from_camera_tkinter.py``` .
2. Convert the dataset into ```python features_extraction_to_csv.py```.
3. To take the detection run ```python detection.py``` .
4. Check the Database by ```python app.py```.for web user interface


## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have any suggestions.


