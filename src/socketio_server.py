"""
基于 Flask-SocketIO 的实时视频流服务器
用于实时姿态分析和视频处理
"""

import cv2
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from src.pose_analyzer import PoseAnalyzer
import json
import logging
import signal
import sys
import yaml
import os
import io
import threading
import time
from pathlib import Path

# 修复 Windows 控制台编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 配置日志

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 全局标志用于优雅关闭
shutdown_flag = False


class FrameThrottler:
    """
    帧节流器 - 跳帧处理机制
    
    如果上一帧还在处理中，则跳过新帧，防止处理队列积压
    这样可以保证：
    1. 服务器始终处理最新的帧
    2. 不会因为处理慢而导致延迟累积
    3. 资源使用更加可控
    """
    
    def __init__(self):
        self.processing = False
        self.lock = threading.Lock()
        self.total_frames = 0
        self.processed_frames = 0
        self.skipped_frames = 0
        self.last_stats_time = time.time()
    
    def should_process(self):
        """
        检查是否应该处理当前帧
        返回 True 表示应该处理，False 表示应该跳过
        """
        with self.lock:
            self.total_frames += 1
            
            if self.processing:
                self.skipped_frames += 1
                # 每100帧打印一次统计
                if self.total_frames % 100 == 0:
                    self._print_stats()
                return False
            
            self.processing = True
            self.processed_frames += 1
            return True
    
    def done_processing(self):
        """标记处理完成"""
        with self.lock:
            self.processing = False
    
    def _print_stats(self):
        """打印性能统计"""
        current_time = time.time()
        elapsed = current_time - self.last_stats_time
        
        if elapsed > 0:
            fps = self.total_frames / elapsed
            process_rate = (self.processed_frames / self.total_frames * 100) if self.total_frames > 0 else 0
            skip_rate = (self.skipped_frames / self.total_frames * 100) if self.total_frames > 0 else 0
            
            logger.info(
                f"[THROTTLER] Total: {self.total_frames} | "
                f"Processed: {self.processed_frames} ({process_rate:.1f}%) | "
                f"Skipped: {self.skipped_frames} ({skip_rate:.1f}%) | "
                f"FPS: {fps:.1f}"
            )
            
            # 🔥 优化：发送网络质量反馈给客户端（用于自适应调整）
            try:
                from flask_socketio import emit
                emit('network_quality', {
                    'skip_rate': skip_rate,
                    'process_rate': process_rate,
                    'fps': fps
                }, broadcast=True)
            except:
                pass  # 如果emit失败，不影响主流程
        
        # 重置统计
        self.total_frames = 0
        self.processed_frames = 0
        self.skipped_frames = 0
        self.last_stats_time = current_time
    
    def get_stats(self):
        """获取当前统计信息"""
        with self.lock:
            return {
                'total_frames': self.total_frames,
                'processed_frames': self.processed_frames,
                'skipped_frames': self.skipped_frames,
                'processing': self.processing
            }


# 全局帧节流器
frame_throttler = FrameThrottler()

# 加载配置文件
def load_config():
    """加载配置文件"""
    config_path = Path('config/config.yaml')
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        print("Warning: Config file not found, using default configuration")
        return {
            'logging': {'enable': True, 'log_dir': 'logs'},
            'ssl': {'enable': False},
            'server': {'host': '0.0.0.0', 'port': 8080, 'https_port': 8443}
        }

config = load_config()

# 初始化 Flask 应用
app = Flask(__name__, 
            static_folder='../assets',
            template_folder='../assets')
app.config['SECRET_KEY'] = 'pose-analysis-secret-key'

# 初始化 SocketIO
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   async_mode='eventlet',
                   ping_timeout=60,
                   ping_interval=25)

# 全局变量
analyzer = None


def init_analyzer():
    """初始化姿态分析器"""
    global analyzer
    if analyzer is None:
        try:
            # 使用 Full 模型（已预装），启用 GPU 加速
            # model_complexity: 0=Lite(需下载), 1=Full(已有), 2=Heavy(已有)
            # enable_gpu: True=启用GPU加速
            # enable_logging: 从配置文件读取
            logging_config = config.get('logging', {})
            enable_logging = logging_config.get('enable', True)
            console_output = logging_config.get('console_output', True)
            record_interval = logging_config.get('record_interval', 1.0)
            
            analyzer = PoseAnalyzer(
                model_complexity=1, 
                enable_gpu=True, 
                enable_logging=enable_logging,
                console_output=console_output,
                record_interval=record_interval
            )
            logger.info("Pose analyzer initialized (Full model + GPU acceleration)")
            if enable_logging:
                logger.info("Logging enabled")
                if console_output:
                    logger.info("Console output enabled (Print Monitor Style)")
        except Exception as e:
            logger.error(f"姿态分析器初始化失败: {e}")
            raise


@app.route('/')
def index():
    """提供主页面"""
    return render_template('demo_socketio.html')


@app.route('/health')
def health():
    """健康检查端点（用于 Docker 等）"""
    try:
        # 检查分析器是否初始化
        if analyzer is None:
            return {'status': 'initializing', 'analyzer': False}, 503
        return {'status': 'healthy', 'analyzer': True}, 200
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500


@app.route('/api/logs/recent')
def get_recent_logs():
    """获取最近的日志记录"""
    try:
        if analyzer and analyzer.logger:
            limit = request.args.get('limit', 100, type=int)
            logs = analyzer.logger.get_recent_logs(limit=limit)
            return jsonify({'status': 'success', 'logs': logs, 'count': len(logs)})
        else:
            return jsonify({'status': 'error', 'message': '日志功能未启用'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/logs/statistics')
def get_statistics():
    """获取统计信息"""
    try:
        if analyzer and analyzer.logger:
            date = request.args.get('date', None)
            stats = analyzer.logger.get_statistics(date=date)
            return jsonify({'status': 'success', 'statistics': stats})
        else:
            return jsonify({'status': 'error', 'message': '日志功能未启用'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/throttler/stats')
def get_throttler_stats():
    """获取跳帧处理统计信息"""
    try:
        stats = frame_throttler.get_stats()
        return jsonify({
            'status': 'success',
            'throttler': stats,
            'description': {
                'total_frames': '总接收帧数',
                'processed_frames': '实际处理帧数',
                'skipped_frames': '跳过帧数',
                'processing': '当前是否正在处理'
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@socketio.on('connect')
def handle_connect():
    """客户端连接事件"""
    logger.info(f"Client connected: {request.sid}")
    emit('server_response', {'status': 'connected', 'message': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开连接事件"""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('video_frame')
def handle_video_frame(data):
    """
    接收并处理视频帧（跳帧处理模式）
    data: {
        'frame': base64编码的图像数据
    }
    
    跳帧逻辑：
    - 如果上一帧还在处理中，则跳过当前帧
    - 这样可以避免处理队列积压，保持实时性
    """
    # 🔥 跳帧检查：如果正在处理，跳过这一帧
    if not frame_throttler.should_process():
        # 静默跳过，不发送任何响应
        return
    
    try:
        # 解码 base64 图像
        frame_data = data.get('frame', '')
        if not frame_data:
            emit('error', {'message': '未收到图像数据'})
            frame_throttler.done_processing()
            return
        
        # 移除 data URL 前缀（如果有）
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        # 解码 base64
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            emit('error', {'message': '图像解码失败'})
            frame_throttler.done_processing()
            return
        
        # 姿态分析
        try:
            processed_frame, pose_data = analyzer.process_frame(frame)
            
            # 编码处理后的图像为 base64
            # 🔥 20 FPS模式：降低JPEG质量到50（应对高帧率的大数据量）
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            processed_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # 发送处理结果
            emit('processed_frame', {
                'frame': f'data:image/jpeg;base64,{processed_b64}',
                'pose_data': pose_data
            })
            
        except Exception as e:
            logger.error(f"姿态分析失败: {e}")
            # 如果分析失败，返回原始图像
            # 🔥 20 FPS模式：降低JPEG质量到50
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            original_b64 = base64.b64encode(buffer).decode('utf-8')
            emit('processed_frame', {
                'frame': f'data:image/jpeg;base64,{original_b64}',
                'pose_data': None,
                'error': str(e)
            })
        
        # 🔥 处理完成，释放锁
        frame_throttler.done_processing()
            
    except Exception as e:
        logger.error(f"处理视频帧时出错: {e}")
        emit('error', {'message': f'处理失败: {str(e)}'})
        # 🔥 确保释放锁
        frame_throttler.done_processing()


@socketio.on('start_stream')
def handle_start_stream():
    """开始视频流"""
    logger.info(f"Client {request.sid} started video stream")
    emit('stream_started', {'status': 'ok'})


@socketio.on('stop_stream')
def handle_stop_stream():
    """停止视频流"""
    logger.info(f"Client {request.sid} stopped video stream")
    emit('stream_stopped', {'status': 'ok'})


def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    global shutdown_flag
    logger.info("\nReceived stop signal, shutting down server...")
    shutdown_flag = True
    # 在 Windows 上强制退出
    sys.exit(0)


def main(host=None, port=None):
    """启动服务器"""
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    # 从配置文件读取服务器配置
    server_config = config.get('server', {})
    if host is None:
        host = server_config.get('host', '0.0.0.0')
    if port is None:
        port = server_config.get('port', 8080)
    
    # 初始化姿态分析器
    try:
        init_analyzer()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return
    
    # SSL 配置
    ssl_config = config.get('ssl', {})
    ssl_enabled = ssl_config.get('enable', False)
    
    if ssl_enabled:
        cert_file = ssl_config.get('cert_file', 'certs/server.crt')
        key_file = ssl_config.get('key_file', 'certs/server.key')
        https_port = server_config.get('https_port', 8443)
        
        # 检查证书文件是否存在
        if not os.path.exists(cert_file) or not os.path.exists(key_file):
            logger.error(f"SSL certificate files not found: {cert_file} or {key_file}")
            logger.info("Tip: Run 'python src/generate_ssl_cert.py' to generate self-signed certificate")
            logger.info("Or set ssl.enable to false in config.yaml")
            return
        
        logger.info(f"Starting SocketIO server (HTTPS) at https://{host}:{https_port}")
        logger.info(f"SSL certificate: {cert_file}")
        logger.info("Press Ctrl+C to stop the server")
        
        try:
            # 运行 HTTPS 服务器
            socketio.run(app, 
                        host=host, 
                        port=https_port, 
                        debug=False, 
                        use_reloader=False,
                        certfile=cert_file,
                        keyfile=key_file)
        except KeyboardInterrupt:
            logger.info("\nServer stopped")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            logger.info("Cleaning up resources...")
    else:
        logger.info(f"Starting SocketIO server (HTTP) at http://{host}:{port}")
        logger.info("Press Ctrl+C to stop the server")
        logger.info("Tip: To enable HTTPS, set ssl.enable: true in config.yaml")
        
        try:
            # 运行 HTTP 服务器
            socketio.run(app, host=host, port=port, debug=False, use_reloader=False)
        except KeyboardInterrupt:
            logger.info("\nServer stopped")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            logger.info("Cleaning up resources...")


if __name__ == "__main__":
    main()

