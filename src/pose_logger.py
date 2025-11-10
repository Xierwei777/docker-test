"""
姿态分析结果记录模块
用于保存和管理分析历史记录
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
import threading
import csv


class PoseLogger:
    def __init__(self, log_dir: str = "logs", max_records: int = 10000, console_output: bool = True, record_interval: float = 1.0):
        """
        初始化姿态日志记录器
        
        Args:
            log_dir: 日志保存目录
            max_records: 单个日志文件最大记录数
            console_output: 是否在控制台实时打印检测结果
            record_interval: 记录间隔（秒），控制日志记录频率
        """
        self.log_dir = log_dir
        self.max_records = max_records
        self.lock = threading.Lock()
        self.console_output = console_output
        self.record_interval = record_interval  # 记录间隔
        self.last_record_time = 0  # 上次记录时间
        self.record_count = 0  # 记录计数器
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 当前日志文件路径
        self.current_date = datetime.now().strftime("%Y%m%d")
        self.json_log_path = os.path.join(log_dir, f"pose_analysis_{self.current_date}.json")
        self.csv_log_path = os.path.join(log_dir, f"pose_analysis_{self.current_date}.csv")
        
        # 初始化CSV文件
        self._init_csv_log()
        
        print(f"[LOG] Pose analysis logger initialized")
        print(f"[LOG] JSON log: {self.json_log_path}")
        print(f"[LOG] CSV log: {self.csv_log_path}")
        print(f"[LOG] Record interval: {record_interval}s (Log recording only, video stream unaffected)")
        if console_output:
            print(f"[LOG] Console output: ENABLED")
            print("=" * 80)
            print("POSTURE DETECTION MONITOR - Real-time Analysis")
            print("=" * 80)
    
    def _init_csv_log(self):
        """初始化CSV日志文件（如果不存在）"""
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    '时间戳', 'ISO时间', '姿态类型', '置信度',
                    '检测到目标', '关键点数量', '备注'
                ])
    
    def log_analysis(self, analysis_result: Dict):
        """
        记录单次分析结果
        
        Args:
            analysis_result: 分析结果字典，应包含:
                - timestamp: 时间戳
                - datetime: ISO格式时间
                - pose: 姿态类型
                - has_target: 是否检测到目标
                - landmarks_count: 关键点数量
                - confidence: 置信度
                - note: 备注信息
        """
        with self.lock:
            try:
                # 检查是否应该记录（基于时间间隔）
                current_time = time.time()
                if current_time - self.last_record_time < self.record_interval:
                    return  # 跳过本次记录
                
                # 更新上次记录时间
                self.last_record_time = current_time
                
                # 增加记录计数
                self.record_count += 1
                
                # 控制台打印台风格输出
                if self.console_output:
                    self._print_console_record(analysis_result)
                
                # 检查是否需要切换日志文件（新的一天）
                current_date = datetime.now().strftime("%Y%m%d")
                if current_date != self.current_date:
                    self.current_date = current_date
                    self.json_log_path = os.path.join(
                        self.log_dir, f"pose_analysis_{self.current_date}.json"
                    )
                    self.csv_log_path = os.path.join(
                        self.log_dir, f"pose_analysis_{self.current_date}.csv"
                    )
                    self._init_csv_log()
                
                # 写入JSON日志（追加模式）
                with open(self.json_log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(analysis_result, ensure_ascii=False) + '\n')
                
                # 写入CSV日志
                with open(self.csv_log_path, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        analysis_result.get('timestamp', ''),
                        analysis_result.get('datetime', ''),
                        analysis_result.get('pose', 'Unknown'),
                        analysis_result.get('confidence', 0.0),
                        '是' if analysis_result.get('has_target', False) else '否',
                        analysis_result.get('landmarks_count', 0),
                        analysis_result.get('note', '')
                    ])
                
            except Exception as e:
                    print(f"[LOG ERROR] Failed to write log: {e}")
    
    def _print_console_record(self, result: Dict):
        """
        在控制台打印检测记录（打印台风格）
        """
        time_str = result.get('datetime', '')
        pose = result.get('pose', 'Unknown')
        has_target = result.get('has_target', False)
        confidence = result.get('confidence', 0.0)
        landmarks = result.get('landmarks_count', 0)
        
        # 根据姿态类型设置状态标记
        if not has_target:
            status = "[ NO TARGET ]"
            pose_display = "No Target"
        else:
            status = "[  DETECT   ]"
            pose_display = pose.upper()
        
        # 格式化输出（类似打印台）
        print(f"#{self.record_count:05d} | {time_str} | {status} | Pose: {pose_display:12s} | Conf: {confidence:.2f} | Points: {landmarks:2d}")
        
        # 每50条记录打印一次分隔线
        if self.record_count % 50 == 0:
            print("-" * 80)
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict]:
        """
        获取最近的日志记录
        
        Args:
            limit: 返回记录数量限制
            
        Returns:
            日志记录列表
        """
        logs = []
        try:
            if os.path.exists(self.json_log_path):
                with open(self.json_log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # 获取最后N条记录
                    for line in lines[-limit:]:
                        try:
                            logs.append(json.loads(line.strip()))
                        except:
                            continue
        except Exception as e:
            print(f"[LOG ERROR] Failed to read log: {e}")
        
        return logs
    
    def get_statistics(self, date: Optional[str] = None) -> Dict:
        """
        获取统计信息
        
        Args:
            date: 日期字符串 (YYYYMMDD)，None表示今天
            
        Returns:
            统计信息字典
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        log_path = os.path.join(self.log_dir, f"pose_analysis_{date}.json")
        
        stats = {
            'total_count': 0,
            'has_target_count': 0,
            'no_target_count': 0,
            'pose_distribution': {},
            'date': date
        }
        
        try:
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            stats['total_count'] += 1
                            
                            if record.get('has_target', False):
                                stats['has_target_count'] += 1
                                pose = record.get('pose', 'Unknown')
                                stats['pose_distribution'][pose] = \
                                    stats['pose_distribution'].get(pose, 0) + 1
                            else:
                                stats['no_target_count'] += 1
                        except:
                            continue
        except Exception as e:
            print(f"[LOG ERROR] Failed to generate statistics: {e}")
        
        return stats
    
    def clear_old_logs(self, days: int = 30):
        """
        清理旧日志文件
        
        Args:
            days: 保留最近N天的日志
        """
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for filename in os.listdir(self.log_dir):
                if filename.startswith('pose_analysis_'):
                    file_path = os.path.join(self.log_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_time < cutoff_date:
                        os.remove(file_path)
                        print(f"[LOG] Deleted old log: {filename}")
        except Exception as e:
            print(f"[LOG ERROR] Failed to clean up logs: {e}")


