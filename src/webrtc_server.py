"""
⚠️ 此文件已弃用 ⚠️

WebRTC 服务器 (webrtc_server.py) 已被 Flask-SocketIO 实现取代。

原因：
- aiortc 在 Windows 上安装困难，需要编译 PyAV
- 依赖复杂，维护成本高
- SocketIO 提供类似功能，安装更简单

新的实时视频流服务：
- 文件：src/socketio_server.py
- 启动：python -m src.socketio_server 或运行 run_socketio.bat
- 文档：docs/socketio_usage.md

如果您仍需要 WebRTC 功能，请参考以下资源：
- https://github.com/aiortc/aiortc
- https://webrtc.org/

================================================================================
以下是原始代码（已不再维护）
================================================================================
"""

import asyncio
import json
import cv2
import numpy as np

# 以下导入已被注释，因为 aiortc 不再是项目依赖
# from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
# from aiohttp import web
# from av import VideoFrame

from aiohttp import web
from src.pose_analyzer import PoseAnalyzer
import base64

# 全局变量
analyzer:PoseAnalyzer = None

pcs = set()

class ProcessedVideoTrack(VideoStreamTrack):
    """处理后的视频轨道，直接返回处理后的帧"""
    
    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0

    async def recv(self):
        try:
            frame = await self.track.recv()
            img = frame.to_ndarray(format="bgr24")
            
            print(f"收到原始帧，尺寸: {img.shape}")
            
            # 姿态分析
            try:
                processed_img, pose_data = analyzer.process_frame(img)
                print(f"姿态分析完成，处理后图像尺寸: {processed_img.shape}")
            except Exception as e:
                print(f"姿态分析失败: {e}")
                # 如果处理失败，返回原始图像
                processed_img = img
            
            # 确保图像尺寸正确
            if processed_img.shape != img.shape:
                print(f"调整图像尺寸从 {processed_img.shape} 到 {img.shape}")
                processed_img = cv2.resize(processed_img, (img.shape[1], img.shape[0]))
            
            # 创建新的VideoFrame
            new_frame = VideoFrame.from_ndarray(processed_img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            
            self.frame_count += 1
            if self.frame_count % 30 == 0:  # 每30帧打印一次
                print(f"处理了 {self.frame_count} 帧")
                
            return new_frame
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
            # 返回原始帧作为fallback
            return frame

async def offer_handler(request):
    """处理WebRTC offer请求"""
    try:
        params = await request.json()
        
        if not params or "sdp" not in params:
            return web.Response(status=400, text="Missing SDP")
        
        sdp = params["sdp"]
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        
        pc = RTCPeerConnection()
        pcs.add(pc)
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print("连接状态:", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)
        
        @pc.on("track")
        def on_track(track):
            print(f"收到轨道: {track.kind}")
            if track.kind == "video":
                print("添加处理后的视频轨道")
                # 添加处理后的视频轨道
                pc.addTrack(ProcessedVideoTrack(track))
        
        # 设置远程描述
        await pc.setRemoteDescription(offer)
        
        # 创建应答
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        print("WebRTC连接建立成功")
        
        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
        
    except Exception as e:
        print(f"处理offer时出错: {e}")
        return web.Response(status=500, text=f"Internal error: {str(e)}")

async def index_handler(request):
    """提供HTML页面"""
    return web.FileResponse('assets/demo_rtc.html')

async def on_shutdown(app):
    """关闭时清理连接"""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

def main():
    app = web.Application()
    app.router.add_post("/offer", offer_handler)
    app.router.add_get("/", index_handler)
    app.router.add_static('/assets', path='./assets', name='assets')
    app.on_shutdown.append(on_shutdown)
    
    # print("启动WebRTC服务器在 http://localhost:8080")
    web.run_app(app, host="0.0.0.0", port=8080)

# if __name__ == "__main__":
#     main()
    
    