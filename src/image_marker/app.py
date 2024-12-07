from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import scipy.io as sio
import numpy as np

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')  # Serve the main HTML page

@app.route('/save_pixels', methods=['POST'])
def save_pixels():
    data = request.get_json()
    marked_pixels = data.get('markedPixels', [])
    width = data.get('width')
    height = data.get('height')

    if not marked_pixels or not width or not height:
        return jsonify({'error': 'Invalid input'}), 400

    # Create a matrix of zeros with the same size as the image
    pixel_matrix = np.zeros((height, width), dtype=np.uint8)

    # Assign classes to pixels based on user input
    for pixel in marked_pixels:
        x, y, class_id = pixel['x'], pixel['y'], pixel['classId']
        pixel_matrix[y, x] = class_id

    # Save the matrix as a .mat file
    output_path = os.path.join(os.getcwd(), 'marked_pixels.mat')
    sio.savemat(output_path, {'pixel_matrix': pixel_matrix})

    return jsonify({'message': f'Pixels saved to {output_path}'}), 200


if __name__ == '__main__':
    app.run(debug=True)