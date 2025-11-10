"""
主服务启动入口
同时启动 HTTP API 服务和 SocketIO 实时视频流服务
"""

import threading
from src.pose_analyzer import PoseAnalyzer

# 创建全局 analyzer 实例
analyzer = PoseAnalyzer()


def start_http_server():
    """启动基于 Flask 的 HTTP API 服务（端口 5000）"""
    from src import api
    api.analyzer = analyzer
    
    print("[INFO] 正在启动 HTTP API 服务...")
    print("      访问地址: http://localhost:5000")
    print("      API 文档: docs/api.md")
    api.app.run(host='0.0.0.0', port=5000, debug=False)


def start_socketio_server():
    """启动基于 Flask-SocketIO 的实时视频流服务（端口 8080）"""
    from src import socketio_server
    socketio_server.analyzer = analyzer
    
    print("[INFO] 正在启动 SocketIO 实时视频流服务...")
    print("      访问地址: http://localhost:8080")
    print("      使用说明: docs/socketio_usage.md")
    socketio_server.main(host='0.0.0.0', port=8080)


if __name__ == '__main__':
    print("=" * 60)
    print("  姿态分析系统 - 服务启动")
    print("=" * 60)
    print()
    
    # 提示选择启动模式
    print("请选择启动模式：")
    print("  1. 仅启动 HTTP API 服务（端口 5000）")
    print("  2. 仅启动 SocketIO 实时视频流服务（端口 8080，推荐）")
    print("  3. 同时启动两个服务")
    print()
    
    choice = input("请输入选项 (1/2/3，默认为 2): ").strip() or "2"
    
    print()
    print("-" * 60)
    
    if choice == "1":
        start_http_server()
    elif choice == "2":
        start_socketio_server()
    elif choice == "3":
        # 使用线程分别运行两个服务
        http_thread = threading.Thread(target=start_http_server, daemon=True)
        http_thread.start()
        
        print()
        print("[INFO] 两个服务已启动：")
        print("      - HTTP API: http://localhost:5000")
        print("      - SocketIO: http://localhost:8080")
        print()
        print("[提示] 按 Ctrl+C 停止服务")
        print("-" * 60)
        
        # 主线程运行 SocketIO 服务
        start_socketio_server()
    else:
        print("[错误] 无效的选项")
        exit(1)


