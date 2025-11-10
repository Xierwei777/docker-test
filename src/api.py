from flask import Flask, request, jsonify, Response
import base64
import io
import numpy as np
from PIL import Image

from src.pose_analyzer import PoseAnalyzer
import cv2

# 使用 assets 目录作为静态资源目录
app = Flask(__name__,
            static_url_path='/static',
            static_folder='./assets',
            root_path='.')
analyzer: PoseAnalyzer = None

def base64_to_image(base64_str: str) -> np.ndarray:
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')
#
# @socketio.on('frame')
# def handle_frame(data):
#     image = base64_to_image(data)
#     processed_frame, pose_data = analyzer.process_frame(image)
#     print(pose_data)
#     _, buffer = cv2.imencode('.jpg', processed_frame)
#     processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
#     emit('result', {'processed_image': processed_image_base64})

@app.route('/')
def index():
    return app.send_static_file('demo.html')  # 默认访问 demo.html


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400

        image = base64_to_image(data['image'])
        processed_frame, pose_data = analyzer.process_frame(image)

        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'landmarks': pose_data['landmarks'],
            'processed_image': processed_image_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 新增：视频流接口
def generate_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 处理帧
        processed_frame, _ = analyzer.process_frame(frame)
        # 编码为 JPEG
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000, debug=False)
#     socketio.run(app, host='0.0.0.0', port=8000)
