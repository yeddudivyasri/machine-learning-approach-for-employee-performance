# Machine Learning Approach for Employee Performance

This project is a web-based machine learning application built with Python (Flask) and LightGBM. It predicts the actual productivity of garment workers based on various operational and team-based input factors.

The model is trained on real-world garment industry data and helps managers and production planners estimate team performance accurately and efficiently.

## Features

- Predicts actual productivity using a trained LightGBM regression model
- Simple form to input relevant parameters like department, SMV, overtime, idle time, style changes, etc.
- Real-time predictions through a clean web interface using HTML, CSS, and JavaScript
- Flask backend handles model training and prediction
- Lightweight and fast with minimal dependencies

## Installation

1. Clone the repository:

   git clone https://github.com/yeddudivyasri/machine-learning-approach-for-employee-performance.git

2. Navigate to the project directory:

   cd machine-learning-approach-for-employee-performance

3. Install the required dependencies:

   pip install -r requirements.txt

4. Run the Flask application:

   python app.py

Once the server starts, open your browser and visit:
http://127.0.0.1:5000/

Input the required values in the form and click Predict to get the actual productivity.

## Dataset

- Filename: garments_worker_productivity.csv
- Contains information about garment factory teams, production metrics, and worker behavior
- Target variable: actual_productivity

## Tech Stack

- Python
- Flask
- LightGBM
- Pandas
- HTML/CSS
- JavaScript

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Author

Divya Sri  
https://github.com/yeddudivyasri
