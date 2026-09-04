from flask import Flask, render_template, request, jsonify, Response, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import json
from datetime import datetime
import csv
from io import BytesIO
import threading
import time

app = Flask(__name__, static_folder='.', template_folder='.')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['RECORD_FOLDER'] = 'recordings'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['RECORD_FOLDER'], exist_ok=True)

# ============ 模型路径 ============
MODEL_PATH = "runs/detect/runs/train/Protected Animal Detection System/weights/best.pt"
model = YOLO(MODEL_PATH)

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

# ============ 检测是否在云环境 ============
IS_CLOUD = os.environ.get('RENDER', False) or os.environ.get('RAILWAY', False) or os.environ.get('CLOUD_ENV', False)

# 本地使用摄像头，云端使用测试视频
if IS_CLOUD:
    IP_CAMERA_URL = "static/sample_video.mp4"
    print("☁️ 云环境模式：使用示例视频")
else:
    IP_CAMERA_URL = 0
    print("💻 本地模式：使用摄像头")

# ============ 创建示例视频（如果不存在） ============
if IS_CLOUD and not os.path.exists("static/sample_video.mp4"):
    os.makedirs("static", exist_ok=True)
    # 创建一个简单的测试视频（如果需要，可以在部署前手动上传一个视频文件）
    print("⚠️ 请上传示例视频到 static/sample_video.mp4")

latest_stats = {
    'total_animals': 0,
    'avg_confidence': 0.0,
    'max_confidence': 0.0,
    'classes_detected': '-',
    'class_counts': {}
}


def generate_combined_stream(url, conf=DEFAULT_CONF, iou=DEFAULT_IOU):
    """生成合成视频流 - 原始画面（左）和检测画面（右）拼合"""
    print("🔍 启动合成视频流")
    print(f"   url={url}, conf={conf}, iou={iou}")

    # 支持本地摄像头（整数）和视频文件（字符串）
    if isinstance(url, int):
        print(f"📷 使用本地摄像头，索引: {url}")
        cap = cv2.VideoCapture(url)
        time.sleep(0.5)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 25)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    elif isinstance(url, str) and url.isdigit():
        url_int = int(url)
        print(f"📷 使用本地摄像头，索引: {url_int}")
        cap = cv2.VideoCapture(url_int)
        time.sleep(0.5)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 25)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        print(f"📷 使用视频源: {url}")
        cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print(f"❌ 无法打开视频源: {url}")
        # 如果打不开，尝试使用本地摄像头作为备选
        print("🔄 尝试使用本地摄像头...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 所有视频源都无法打开")
            return

    print("✅ 视频源连接成功")

    ret, test_frame = cap.read()
    if ret:
        h, w = test_frame.shape[:2]
        print(f"✅ 成功读取第一帧，尺寸: {w}x{h}")
    else:
        print("⚠️ 警告：第一帧读取失败，但继续尝试...")

    target_fps = 15
    frame_delay = 1.0 / target_fps
    frame_counter = 0

    while True:
        start_time = time.time()

        success, frame = cap.read()
        if not success:
            print("⚠️ 读取视频帧失败")
            # 如果是视频文件，重新播放
            if isinstance(url, str) and not url.isdigit():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            # 如果是摄像头，尝试重连
            if isinstance(url, int):
                cap = cv2.VideoCapture(url)
            elif isinstance(url, str) and url.isdigit():
                cap = cv2.VideoCapture(int(url))
            else:
                cap = cv2.VideoCapture(url)
            time.sleep(0.5)
            continue

        frame_counter += 1

        # ===== 左半边：原始画面 =====
        original = frame.copy()

        # ===== 右半边：检测画面 =====
        try:
            # 每帧都检测
            results = model(frame, conf=conf, iou=iou)
            detected = results[0].plot()

            boxes = results[0].boxes
            num_animals = int(len(boxes))

            if num_animals > 0:
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)

                latest_stats['total_animals'] = num_animals
                latest_stats['avg_confidence'] = float(np.mean(confidences))
                latest_stats['max_confidence'] = float(np.max(confidences))

                class_counts = {}
                class_names = []
                for cls in class_ids:
                    class_name = model.names[int(cls)]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    if class_name not in class_names:
                        class_names.append(class_name)

                latest_stats['classes_detected'] = ', '.join(class_names)
                latest_stats['class_counts'] = class_counts
            else:
                latest_stats['total_animals'] = 0
                latest_stats['avg_confidence'] = 0.0
                latest_stats['max_confidence'] = 0.0
                latest_stats['classes_detected'] = '-'
                latest_stats['class_counts'] = {}

        except Exception as e:
            print(f"❌ 检测出错: {e}")
            detected = frame.copy()
            cv2.putText(detected, "Detection Error", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ===== 拼合左右画面 =====
        if original.shape != detected.shape:
            detected = cv2.resize(detected, (original.shape[1], original.shape[0]))

        combined = np.hstack([original, detected])
        cv2.line(combined, (original.shape[1], 0), (original.shape[1], combined.shape[0]), (255, 255, 255), 2)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        ret, buffer = cv2.imencode('.jpg', combined, encode_param)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        elapsed = time.time() - start_time
        sleep_time = max(0, frame_delay - elapsed)
        time.sleep(sleep_time)

    cap.release()
    print("📷 视频源已断开")


def analyze_results(results):
    boxes = results[0].boxes
    num_animals = int(len(boxes))
    if num_animals == 0:
        return {'count': 0, 'avg_confidence': 0, 'min_confidence': 0, 'max_confidence': 0, 'class_counts': {},
                'detections': []}
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    detections = []
    class_counts = {}
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(val) for val in box.xyxy[0].cpu().numpy()]
        conf = float(box.conf[0].item())
        cls = int(box.cls[0].item())
        class_name = model.names[cls]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        detections.append({'id': i + 1, 'confidence': round(float(conf), 4), 'class': class_name,
                           'bbox': {'x1': round(float(x1), 2), 'y1': round(float(y1), 2), 'x2': round(float(x2), 2),
                                    'y2': round(float(y2), 2)}})
    return {'count': int(num_animals), 'avg_confidence': round(float(np.mean(confidences)), 4),
            'min_confidence': round(float(np.min(confidences)), 4),
            'max_confidence': round(float(np.max(confidences)), 4), 'class_counts': class_counts,
            'detections': detections}


@app.route('/')
def index():
    return render_template('system.html')


@app.route('/combined_feed')
def combined_feed():
    print("=" * 50)
    print("📹 /combined_feed 被访问")
    url_param = request.args.get('url')
    conf_param = request.args.get('conf')
    iou_param = request.args.get('iou')
    if url_param is not None:
        if isinstance(url_param, str) and url_param.isdigit():
            url = int(url_param)
        else:
            url = url_param
    else:
        url = IP_CAMERA_URL
    if conf_param is not None:
        conf = float(conf_param)
    else:
        conf = DEFAULT_CONF
    if iou_param is not None:
        iou = float(iou_param)
    else:
        iou = DEFAULT_IOU
    print(f"   最终参数: url={url}, conf={conf}, iou={iou}")
    print("=" * 50)
    return Response(generate_combined_stream(url, conf=conf, iou=iou),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    return jsonify(latest_stats)


@app.route('/screenshot')
def screenshot():
    try:
        # 支持本地摄像头和视频文件
        if isinstance(IP_CAMERA_URL, int):
            cap = cv2.VideoCapture(IP_CAMERA_URL)
        elif isinstance(IP_CAMERA_URL, str) and IP_CAMERA_URL.isdigit():
            cap = cv2.VideoCapture(int(IP_CAMERA_URL))
        else:
            cap = cv2.VideoCapture(IP_CAMERA_URL)
        if not cap.isOpened():
            return jsonify({'error': '无法打开视频源'}), 500
        success, frame = cap.read()
        cap.release()
        if not success:
            return jsonify({'error': '截屏失败'}), 500
        results = model(frame, conf=DEFAULT_CONF, iou=DEFAULT_IOU)
        detected = results[0].plot()
        h, w = frame.shape[:2]
        original = frame.copy()
        if detected.shape != original.shape:
            detected = cv2.resize(detected, (w, h))
        combined = np.hstack([original, detected])
        cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', combined)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'success': True, 'image': f'data:image/jpeg;base64,{img_base64}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect_single', methods=['POST'])
def detect_single():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        file = request.files['file']
        conf = float(request.form.get('conf', DEFAULT_CONF))
        iou = float(request.form.get('iou', DEFAULT_IOU))
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'success': False, 'error': 'Invalid image'}), 400
        results = model(img, conf=conf, iou=iou)
        annotated_img = results[0].plot()
        stats = analyze_results(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f'result_{timestamp}.jpg')
        cv2.imwrite(result_path, annotated_img)
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify(
            {'success': True, 'result_image': f'data:image/jpeg;base64,{img_base64}', 'detections': stats['detections'],
             'animal_count': stats['count'], 'avg_confidence': stats['avg_confidence'],
             'class_counts': stats['class_counts'], 'result_file': result_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/detect', methods=['POST'])
def detect_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['file']
        conf = float(request.args.get('conf', DEFAULT_CONF))
        iou = float(request.args.get('iou', DEFAULT_IOU))
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400
        results = model(img, conf=conf, iou=iou)
        annotated_img = results[0].plot()
        stats = analyze_results(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f'result_{timestamp}.jpg')
        cv2.imwrite(result_path, annotated_img)
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify(
            {'success': True, 'result_image': f'data:image/jpeg;base64,{img_base64}', 'animal_count': stats['count'],
             'avg_confidence': stats['avg_confidence'], 'class_counts': stats['class_counts'],
             'result_file': result_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/batch_detect', methods=['POST'])
def detect_batch():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        files = request.files.getlist('files')
        conf = float(request.args.get('conf', DEFAULT_CONF))
        iou = float(request.args.get('iou', DEFAULT_IOU))
        results = []
        total_animals = 0
        all_class_counts = {}
        for file in files:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            detection_results = model(img, conf=conf, iou=iou)
            stats = analyze_results(detection_results)
            for class_name, count in stats['class_counts'].items():
                all_class_counts[class_name] = all_class_counts.get(class_name, 0) + count
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(app.config['RESULTS_FOLDER'], f'batch_{timestamp}_{file.filename}')
            annotated_img = detection_results[0].plot()
            cv2.imwrite(result_path, annotated_img)
            _, buffer = cv2.imencode('.jpg', annotated_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            results.append(
                {'filename': file.filename, 'animal_count': stats['count'], 'avg_confidence': stats['avg_confidence'],
                 'class_counts': stats['class_counts'], 'result_image': f'data:image/jpeg;base64,{img_base64}',
                 'result_file': result_path})
            total_animals += stats['count']
        return jsonify(
            {'success': True, 'results': results, 'total_images': len(results), 'total_animals': total_animals,
             'overall_class_counts': all_class_counts,
             'avg_animals_per_image': round(total_animals / len(results), 2) if results else 0})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/export_csv', methods=['POST'])
def export_csv():
    try:
        data = request.json.get('data', [])
        output = BytesIO()
        writer = csv.writer(output)
        writer.writerow(['Filename', 'Animal Count', 'Avg Confidence', 'Result File'])
        for item in data:
            writer.writerow([item['filename'], item['animal_count'], item['avg_confidence'], item['result_file']])
        output.seek(0)
        return send_file(output, mimetype='text/csv', as_attachment=True,
                         download_name=f'animal_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/animal/<path:filename>')
def serve_animal_image(filename):
    try:
        animal_folder = os.path.join('animal', filename)
        if os.path.exists(animal_folder):
            import glob
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
            images = []
            for ext in image_extensions:
                pattern = os.path.join(animal_folder, ext)
                images.extend(glob.glob(pattern))
                pattern = os.path.join(animal_folder, ext.upper())
                images.extend(glob.glob(pattern))
            if images:
                first_image = sorted(images)[0]
                return send_file(first_image)
        return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_animal_images')
def get_animal_images():
    try:
        import glob
        folder = request.args.get('folder', '')
        limit = int(request.args.get('limit', 20))
        if not folder:
            return jsonify({'success': False, 'error': 'No folder specified'})
        folder_path = os.path.join('animal', folder)
        if not os.path.exists(folder_path):
            return jsonify({'success': False, 'error': f'Folder not found: {folder_path}'})
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        images = []
        for ext in image_extensions:
            pattern = os.path.join(folder_path, ext)
            images.extend(glob.glob(pattern))
            pattern = os.path.join(folder_path, ext.upper())
            images.extend(glob.glob(pattern))
        print(f"Found {len(images)} images in {folder_path}")
        image_data = []
        for img_path in sorted(images)[:limit]:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    _, buffer = cv2.imencode('.jpg', img)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    image_data.append({'src': f'data:image/jpeg;base64,{img_base64}'})
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
        return jsonify({'success': True, 'images': image_data})
    except Exception as e:
        print(f"Error in get_animal_images: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/test_camera')
def test_camera():
    if IS_CLOUD:
        return jsonify({'status': 'info', 'message': '云环境模式，使用示例视频'})
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            return jsonify({'status': 'ok', 'message': '摄像头工作正常', 'frame_size': frame.shape})
        else:
            return jsonify({'status': 'error', 'message': '摄像头打开但无法读取画面'})
    else:
        return jsonify({'status': 'error', 'message': '摄像头无法打开'})


@app.route('/health')
def health():
    """健康检查端点"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'cloud_mode': IS_CLOUD,
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🦁 保护动物智能检测系统启动中...")
    print("=" * 60)
    print(f"📷 视频源: {IP_CAMERA_URL}")
    print(f"☁️ 云模式: {IS_CLOUD}")
    print(f"🌐 访问地址: http://localhost:5000")
    print("=" * 60)

    print("🔄 正在加载模型...")
    try:
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy_img, conf=DEFAULT_CONF, iou=DEFAULT_IOU)
        print("✅ 模型加载完成")
    except Exception as e:
        print(f"⚠️ 模型预热失败: {e}")
        print("将继续启动，但首次检测可能较慢")

    print("=" * 60)
    print("🚀 服务器已启动，按 Ctrl+C 停止")
    print("=" * 60)

    # 部署时使用 gunicorn，本地运行使用 debug 模式
    # 如果使用 gunicorn 启动，不会执行这里
    app.run(debug=True, host='0.0.0.0', port=5000)