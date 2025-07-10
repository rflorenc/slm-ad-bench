#!/usr/bin/env python
"""
System Resource Monitoring Module
Integrates psutil, nvidia-ml-py, and power estimation
"""

import os
import time
import json
import csv
import threading
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - system monitoring will be limited")

try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False
    logging.warning("pynvml not available - GPU monitoring disabled")

# PowerAPI alternative - simple CPU-based power estimation
try:
    import cpuinfo
    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Data class for storing system metrics at a point in time"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    gpu_count: int
    gpu_utilization: List[float]
    gpu_memory_used: List[float]
    gpu_memory_total: List[float]
    gpu_temperature: List[float]
    gpu_power_draw: List[float]
    estimated_cpu_power: float
    estimated_gpu_power: float
    total_estimated_power: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float

class PowerEstimator:
    """Simple power estimation based on CPU/GPU utilization"""
    
    def __init__(self):
        self.cpu_tdp = self._estimate_cpu_tdp()
        self.baseline_power = 50.0  # Estimated system baseline power (watts)
        
    def _estimate_cpu_tdp(self) -> float:
        """Estimate CPU TDP based on CPU info"""
        if not CPUINFO_AVAILABLE:
            return 95.0  # Default TDP
            
        try:
            info = cpuinfo.get_cpu_info()
            cpu_name = info.get('brand_raw', '').lower()
            
            # Simple TDP estimation based on CPU family
            if 'i9' in cpu_name or 'ryzen 9' in cpu_name:
                return 125.0
            elif 'i7' in cpu_name or 'ryzen 7' in cpu_name:
                return 95.0
            elif 'i5' in cpu_name or 'ryzen 5' in cpu_name:
                return 65.0
            elif 'xeon' in cpu_name or 'epyc' in cpu_name:
                return 150.0
            else:
                return 95.0
        except:
            return 95.0
    
    def estimate_cpu_power(self, cpu_percent: float) -> float:
        """Estimate CPU power consumption based on utilization"""
        # Simple linear model: idle power + (utilization * max_power)
        idle_power = self.cpu_tdp * 0.3  # 30% of TDP at idle
        active_power = self.cpu_tdp * 0.7 * (cpu_percent / 100.0)
        return idle_power + active_power
    
    def estimate_gpu_power(self, gpu_utilizations: List[float], gpu_power_draws: List[float]) -> float:
        """Estimate total GPU power consumption"""
        if gpu_power_draws and any(p > 0 for p in gpu_power_draws):
            # Use actual power readings if available
            return sum(gpu_power_draws)
        elif gpu_utilizations:
            return sum(250.0 * (util / 100.0) for util in gpu_utilizations)
        else:
            return 0.0

class ComprehensiveSystemMonitor:
    """Comprehensive system resource monitor matching original functionality"""
    
    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval = sample_interval
        self.power_estimator = PowerEstimator()
        self.metrics_history: List[SystemMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
        
        if NVIDIA_ML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"Initialized NVIDIA monitoring for {self.gpu_count} GPUs")
            except Exception as e:
                logger.warning(f"Failed to initialize NVIDIA monitoring: {e}")
                self.gpu_count = 0
        else:
            self.gpu_count = 0
        
        self._baseline_disk_io = self._get_disk_io()
        self._baseline_network = self._get_network_io()
        
        logger.info(f"System monitoring initialized. GPU count: {self.gpu_count}")
        logger.info(f"Estimated CPU TDP: {self.power_estimator.cpu_tdp}W")
    
    def _get_disk_io(self) -> Tuple[float, float]:
        """Get current disk I/O in MB"""
        if not PSUTIL_AVAILABLE:
            return 0.0, 0.0
        try:
            io = psutil.disk_io_counters()
            return io.read_bytes / (1024**2), io.write_bytes / (1024**2)
        except:
            return 0.0, 0.0
    
    def _get_network_io(self) -> Tuple[float, float]:
        """Get current network I/O in MB"""
        if not PSUTIL_AVAILABLE:
            return 0.0, 0.0
        try:
            io = psutil.net_io_counters()
            return io.bytes_sent / (1024**2), io.bytes_recv / (1024**2)
        except:
            return 0.0, 0.0
    
    def _collect_cpu_memory_metrics(self) -> Tuple[float, float, float, float]:
        """Collect CPU and memory metrics"""
        if not PSUTIL_AVAILABLE:
            return 5.0, 8.0, 2.5, 12.0  # Default values
        
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            return cpu_percent, memory.percent, memory_used_gb, memory_available_gb
        except Exception as e:
            logger.error(f"Error collecting CPU/memory metrics: {e}")
            return 5.0, 8.0, 2.5, 12.0
    
    def _collect_gpu_metrics(self) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """Collect GPU metrics for all available GPUs"""
        if not NVIDIA_ML_AVAILABLE or self.gpu_count == 0:
            return [], [], [], [], []
        
        utilizations, memory_used, memory_total, temperatures, power_draws = [], [], [], [], []
        
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilizations.append(float(util.gpu))
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used.append(mem_info.used / (1024**3))  # GB
                memory_total.append(mem_info.total / (1024**3))  # GB
                
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    temperatures.append(float(temp))
                except:
                    temperatures.append(0.0)
                
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                    power_draws.append(power)
                except:
                    power_draws.append(0.0)
        
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
        
        return utilizations, memory_used, memory_total, temperatures, power_draws
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect all system metrics at current point in time"""
        timestamp = time.time()
        
        cpu_percent, memory_percent, memory_used_gb, memory_available_gb = self._collect_cpu_memory_metrics()
        
        gpu_utilizations, gpu_memory_used, gpu_memory_total, gpu_temperatures, gpu_power_draws = self._collect_gpu_metrics()
        
        current_disk_read, current_disk_write = self._get_disk_io()
        current_net_sent, current_net_recv = self._get_network_io()
        
        disk_io_read_mb = current_disk_read - self._baseline_disk_io[0]
        disk_io_write_mb = current_disk_write - self._baseline_disk_io[1]
        network_sent_mb = current_net_sent - self._baseline_network[0]
        network_recv_mb = current_net_recv - self._baseline_network[1]
        
        estimated_cpu_power = self.power_estimator.estimate_cpu_power(cpu_percent)
        estimated_gpu_power = self.power_estimator.estimate_gpu_power(gpu_utilizations, gpu_power_draws)
        total_estimated_power = self.power_estimator.baseline_power + estimated_cpu_power + estimated_gpu_power
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_available_gb=memory_available_gb,
            gpu_count=self.gpu_count,
            gpu_utilization=gpu_utilizations,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_temperature=gpu_temperatures,
            gpu_power_draw=gpu_power_draws,
            estimated_cpu_power=estimated_cpu_power,
            estimated_gpu_power=estimated_gpu_power,
            total_estimated_power=total_estimated_power,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb
        )
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        logger.info("Started background system monitoring")
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sample_interval)
    
    def start_monitoring(self, operation_name: str = "unknown"):
        """Start background monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics_history.clear()
        
        self._baseline_disk_io = self._get_disk_io()
        self._baseline_network = self._get_network_io()
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Starting monitoring: {operation_name}")
    
    def stop_monitoring(self, operation_name: str = "unknown"):
        """Stop monitoring and return collected metrics"""
        if not self.monitoring:
            return []
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        metrics_count = len(self.metrics_history)
        duration = self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp if metrics_count > 0 else 0.0
        
        logger.info(f"Stopped monitoring. Collected {metrics_count} samples")
        logger.info(f"Completed monitoring: {operation_name} (duration: {duration:.2f}s)")
        
        return self.metrics_history.copy()
    
    def save_metrics(self, metrics: List[SystemMetrics], output_dir: str, operation_name: str):
        """Save metrics to CSV and create plots"""
        if not metrics:
            logger.warning("No metrics to save")
            return {}
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        records = []
        for m in metrics:
            record = asdict(m)
            for i, util in enumerate(m.gpu_utilization):
                record[f'gpu_{i}_utilization'] = util
            for i, mem in enumerate(m.gpu_memory_used):
                record[f'gpu_{i}_memory_used'] = mem
            for i, temp in enumerate(m.gpu_temperature):
                record[f'gpu_{i}_temperature'] = temp
            for i, power in enumerate(m.gpu_power_draw):
                record[f'gpu_{i}_power_draw'] = power
            
            for key in ['gpu_utilization', 'gpu_memory_used', 'gpu_memory_total', 'gpu_temperature', 'gpu_power_draw']:
                record.pop(key, None)
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        csv_file = output_dir / f"system_metrics_{operation_name}_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved {len(metrics)} metrics records to {csv_file}")
        
        plot_files = self._create_plots(df, output_dir, operation_name, timestamp)
        
        summary = self._calculate_summary_stats(metrics, operation_name)
        
        summary_file = output_dir / f"system_summary_{operation_name}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("System monitoring completed for: " + operation_name)
        
        return {
            'csv_file': str(csv_file),
            'summary_file': str(summary_file),
            'plot_files': plot_files,
            'summary': summary
        }
    
    def _create_plots(self, df: pd.DataFrame, output_dir: Path, operation_name: str, timestamp: str) -> Dict[str, str]:
        """Create monitoring plots"""
        plot_files = {}
        
        try:
            plt.style.use('default')
            
            # CPU/Memory plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            ax1.plot(df['timestamp'] - df['timestamp'].iloc[0], df['cpu_percent'], 'b-', label='CPU %')
            ax1.plot(df['timestamp'] - df['timestamp'].iloc[0], df['memory_percent'], 'r-', label='Memory %')
            ax1.set_ylabel('Utilization %')
            ax1.set_title('CPU and Memory Utilization')
            ax1.legend()
            ax1.grid(True)
            
            ax2.plot(df['timestamp'] - df['timestamp'].iloc[0], df['memory_used_gb'], 'g-', label='Memory Used (GB)')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Memory (GB)')
            ax2.set_title('Memory Usage')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            cpu_mem_file = output_dir / f"{operation_name}_{timestamp}_cpu_memory_metrics.png"
            plt.savefig(cpu_mem_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            plot_files['cpu_memory'] = str(cpu_mem_file)
            logger.info(f"Saved CPU/Memory plot to {cpu_mem_file}")
            
            if any(col.startswith('gpu_') for col in df.columns):
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                gpu_util_cols = [col for col in df.columns if col.endswith('_utilization')]
                for col in gpu_util_cols:
                    gpu_id = col.split('_')[1]
                    ax1.plot(df['timestamp'] - df['timestamp'].iloc[0], df[col], label=f'GPU {gpu_id}')
                
                ax1.set_ylabel('GPU Utilization %')
                ax1.set_title('GPU Utilization')
                ax1.legend()
                ax1.grid(True)
                
                gpu_mem_cols = [col for col in df.columns if col.endswith('_memory_used')]
                for col in gpu_mem_cols:
                    gpu_id = col.split('_')[1]
                    ax2.plot(df['timestamp'] - df['timestamp'].iloc[0], df[col], label=f'GPU {gpu_id}')
                
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel('GPU Memory (GB)')
                ax2.set_title('GPU Memory Usage')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                gpu_file = output_dir / f"{operation_name}_{timestamp}_gpu_metrics.png"
                plt.savefig(gpu_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                plot_files['gpu'] = str(gpu_file)
                logger.info(f"Saved GPU plot to {gpu_file}")
            
            # I/O plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
            
            ax1.plot(df['timestamp'] - df['timestamp'].iloc[0], df['disk_io_read_mb'], 'b-', label='Disk Read')
            ax1.plot(df['timestamp'] - df['timestamp'].iloc[0], df['disk_io_write_mb'], 'r-', label='Disk Write')
            ax1.set_ylabel('Disk I/O (MB)')
            ax1.set_title('Disk I/O')
            ax1.legend()
            ax1.grid(True)
            
            ax2.plot(df['timestamp'] - df['timestamp'].iloc[0], df['network_sent_mb'], 'g-', label='Network Sent')
            ax2.plot(df['timestamp'] - df['timestamp'].iloc[0], df['network_recv_mb'], 'm-', label='Network Received')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Network I/O (MB)')
            ax2.set_title('Network I/O')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            io_file = output_dir / f"{operation_name}_{timestamp}_io_metrics.png"
            plt.savefig(io_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            plot_files['io'] = str(io_file)
            logger.info(f"Saved I/O plot to {io_file}")
            
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
        
        return plot_files
    
    def _calculate_summary_stats(self, metrics: List[SystemMetrics], operation_name: str) -> Dict:
        """Calculate summary statistics from metrics"""
        if not metrics:
            return {}
        
        duration = metrics[-1].timestamp - metrics[0].timestamp
        
        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_used_gb for m in metrics]
        power_values = [m.total_estimated_power for m in metrics]
        
        summary = {
            "operation": operation_name,
            "duration_seconds": duration,
            "sample_count": len(metrics),
            "start_time": metrics[0].timestamp,
            "end_time": metrics[-1].timestamp,
            
            # CPU stats
            "cpu_percent_mean": np.mean(cpu_values),
            "cpu_percent_max": np.max(cpu_values),
            "cpu_percent_min": np.min(cpu_values),
            "cpu_percent_std": np.std(cpu_values),
            
            # Memory stats
            "memory_used_gb_mean": np.mean(memory_values),
            "memory_used_gb_max": np.max(memory_values),
            "memory_used_gb_min": np.min(memory_values),
            "memory_used_gb_std": np.std(memory_values),
            
            # Power stats
            "total_estimated_power_mean": np.mean(power_values),
            "total_estimated_power_max": np.max(power_values),
            "total_estimated_power_min": np.min(power_values),
            "total_estimated_power_std": np.std(power_values),
            
            "gpu_count": metrics[0].gpu_count,
        }
        
        if metrics[0].gpu_count > 0:
            for i in range(metrics[0].gpu_count):
                gpu_utils = [m.gpu_utilization[i] if i < len(m.gpu_utilization) else 0.0 for m in metrics]
                gpu_mems = [m.gpu_memory_used[i] if i < len(m.gpu_memory_used) else 0.0 for m in metrics]
                
                summary[f"gpu_{i}_utilization_mean"] = np.mean(gpu_utils)
                summary[f"gpu_{i}_utilization_max"] = np.max(gpu_utils)
                summary[f"gpu_{i}_memory_used_mean"] = np.mean(gpu_mems)
                summary[f"gpu_{i}_memory_used_max"] = np.max(gpu_mems)
        
        return summary

@contextmanager
def monitor_operation(operation_name: str, output_dir: str, 
                     save_csv: bool = True, create_plots: bool = True):
    """
    Context manager for monitoring system resources during an operation
    """
    monitor = ComprehensiveSystemMonitor()
    
    try:
        monitor.start_monitoring(operation_name)
        yield monitor
    finally:
        metrics = monitor.stop_monitoring(operation_name)
        
        if save_csv or create_plots:
            monitor.save_metrics(metrics, output_dir, operation_name)