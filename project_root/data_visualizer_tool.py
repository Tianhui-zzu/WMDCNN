import os
import io
import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import matplotlib

matplotlib.use('Agg')  # 使用后台渲染引擎，防止弹出数百个独立图表窗口导致卡死
from matplotlib.figure import Figure

# ==========================================
# ⚙️ 核心配置区域 (参数化暴露)
# ==========================================
# 归一化算法选择
# 可选参数:
#   None       : (null) 不使用任何归一化，使用原始数值
#   'min-max'  : 最小最大归一化，将所有波形缩放到 0~1 的区间
#   'z-score'  : 标准化，将数据转化为均值为0，标准差为1的分布
NORMALIZATION_METHOD = None

# 数据解析配置
HEADER_ROW = None  # CSV无表头填 None
INDEX_COL_EXISTS = True  # 第1列是否是时间步/序号？(是的话不画该列)

# 界面显示配置
PLOTS_PER_ROW = 4  # 界面每一行显示的图表个数
FIG_SIZE = (3.5, 2.5)  # 单个图表的尺寸 (宽, 高) - 英寸


# ==========================================


class DataVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"传感器数据可视化检查工具 (归一化: {NORMALIZATION_METHOD})")
        self.root.geometry("1200x800")

        # 顶层控制面板
        self.top_frame = tk.Frame(self.root, pady=10)
        self.top_frame.pack(side="top", fill="x")

        self.btn_select = tk.Button(self.top_frame, text="📁 选择数据存放文件夹", command=self.ask_directory,
                                    font=("Arial", 12))
        self.btn_select.pack(side="left", padx=20)

        self.status_label = tk.Label(self.top_frame, text="请点击左侧按钮选择包含 CSV 数据的目录...", fg="blue",
                                     font=("Arial", 11))
        self.status_label.pack(side="left", padx=20)

        # 滚动显示区域
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # 绑定鼠标滚轮 (兼容 Windows, Linux, macOS)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")

    def ask_directory(self):
        """弹出窗口供用户选择路径"""
        directory = filedialog.askdirectory(title="选择增强后的数据文件夹")
        if directory:
            self.load_and_render_data(directory)

    def apply_normalization(self, data, method):
        """执行归一化计算"""
        if method is None:
            return data

        data = np.array(data, dtype=float)

        if method == 'min-max':
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)
            ptp = max_val - min_val
            ptp[ptp == 0] = 1.0  # 防除零
            return (data - min_val) / ptp

        elif method == 'z-score':
            mean_val = np.mean(data, axis=0)
            std_val = np.std(data, axis=0)
            std_val[std_val == 0] = 1.0  # 防除零
            return (data - mean_val) / std_val

        return data

    def load_and_render_data(self, directory):
        """读取数据并渲染图表"""
        csv_files = sorted([f for f in os.listdir(directory) if f.lower().endswith('.csv')])
        if not csv_files:
            self.status_label.config(text=f"错误: 在 {directory} 中没有找到CSV文件", fg="red")
            return

        # 清理先前的图表
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self.status_label.config(text=f"找到了 {len(csv_files)} 个文件，正在渲染中，请稍候...", fg="orange")
        self.root.update()

        # 遍历所有数据开始画图
        for i, filename in enumerate(csv_files):
            filepath = os.path.join(directory, filename)

            try:
                df = pd.read_csv(filepath, header=HEADER_ROW)

                # 剥离索引列
                if INDEX_COL_EXISTS:
                    sensor_data = df.iloc[:, 1:].values
                else:
                    sensor_data = df.values

                # 应用归一化
                sensor_data = self.apply_normalization(sensor_data, NORMALIZATION_METHOD)

                # 渲染 Matplotlib 图表
                fig = Figure(figsize=FIG_SIZE, dpi=80)
                ax = fig.add_subplot(111)
                ax.plot(sensor_data)

                ax.set_title(filename, fontsize=9)
                ax.tick_params(axis='both', which='major', labelsize=7)
                fig.tight_layout()

                # 存入内存缓冲区转化为 Tkinter 图像
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf)
                tk_img = ImageTk.PhotoImage(img)
                fig.clf()  # 释放内存

                # 布置到网格中
                row = i // PLOTS_PER_ROW
                col = i % PLOTS_PER_ROW
                lbl = tk.Label(self.scrollable_frame, image=tk_img)
                lbl.image = tk_img  # 防止被垃圾回收
                lbl.grid(row=row, column=col, padx=5, pady=5)

                # 刷新 UI 进度
                if (i + 1) % 10 == 0 or (i + 1) == len(csv_files):
                    self.status_label.config(text=f"正在极速渲染... 进度: {i + 1} / {len(csv_files)}")
                    self.root.update()

            except Exception as e:
                print(f"[!] 读取或渲染 {filename} 时出错: {e}")

        self.status_label.config(text=f"✅ 渲染完成！共成功展示 {len(csv_files)} 个图表，请通过鼠标滚轮浏览。", fg="green")


if __name__ == "__main__":
    # 创建主窗口并在启动时自动弹出选择目录对话框
    root = tk.Tk()
    app = DataVisualizerApp(root)

    # 延迟 100 毫秒后自动弹出选择路径窗口
    root.after(100, app.ask_directory)

    root.mainloop()