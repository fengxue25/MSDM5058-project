"""
Configuration settings for MSDM5058 Project I
Centralized configuration for analysis parameters
"""

import os
from datetime import datetime

# Analysis Parameters
class AnalysisConfig:
    # Output settings
    OUTPUT_DIR = "output"
    FIGURE_FORMAT = "png"
    FIGURE_DPI = 300
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    
    # Plotting style
    # PLOT_STYLE = "seaborn-v0_8"
    PLOT_STYLE = "seaborn-v0_8-white"
    FIGURE_SIZE = (12, 8)
    FONT_SIZE = 12
    GRID_ALPHA = 0.3

    # RC dictionary
    CUSTOM_RC = {
        "axes.facecolor": "white",      # 无背景色
        "axes.edgecolor": "black",      # 边框颜色
        "axes.linewidth": 1,            # 边框粗细
        "axes.grid": False,             # 禁用网格
        "grid.color": "none",           # 彻底消除网格残留
        "xtick.direction": "in",        # 刻度线向内
        "ytick.direction": "in"
    }
    
    # Color schemes
    COLOR_PALETTE = {
        'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'secondary': ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        'background': '#ffffff'
    }

    @classmethod
    def get_output_directory(cls):
        """Get the output directory path with timestamp"""
        timestamp = datetime.now().strftime(cls.TIMESTAMP_FORMAT)
        return os.path.join(cls.OUTPUT_DIR, timestamp)
    # 生成带时间戳的输出根目录，例如 output/20250320_143022。每次运行都会创建新目录，防止覆盖之前的结果。
    
    
    @classmethod
    def get_directories(cls):
        """Create and return all necessary directories"""
        base_dir = cls.get_output_directory()
        directories = [
            base_dir,
            os.path.join(base_dir, "figures"),
            os.path.join(base_dir, "results"),
            os.path.join(base_dir, "data")
        ]

        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)

        return directories
