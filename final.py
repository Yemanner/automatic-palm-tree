import os
import sys
import subprocess  # 用于调用 ffmpeg 命令行工具
import torch
import cv2
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QLineEdit, QFileDialog,
    QHBoxLayout, QListWidgetItem, QMessageBox, QFrame, QScrollArea, QAbstractItemView,
    QListWidget, QSlider
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QUrl, QSize, QTime
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

# ------------------------------------------------------
# 1) 与图像匹配相关的自定义控件、CLIP 加载与相似度函数
# ------------------------------------------------------

########################################
# 自定义的列表控件，用于拖拽排序
########################################
class MatchedListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 启用内部拖拽排序
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)

    def dropEvent(self, event):
        """
        当用户拖拽移动列表项完成后，需要重新为每个列表项调用 setItemWidget，
        以恢复自定义小部件与列表项的绑定关系。
        """
        super().dropEvent(event)
        # 重新绑定小部件
        for i in range(self.count()):
            item = self.item(i)
            widget = item.data(Qt.UserRole)
            if widget is not None:
                self.setItemWidget(item, widget)
                item.setSizeHint(widget.sizeHint())


########################################
# 自定义小部件，用于显示缩略图 + 删除按钮
########################################
class MatchedItemWidget(QWidget):
    def __init__(self, image_path, remove_callback, parent=None):
        """
        :param image_path: 图片路径
        :param remove_callback: 删除回调函数，形如 remove_matched_item(widget)
        :param parent: 父控件
        """
        super().__init__(parent)
        self.image_path = image_path  # 保存图片路径

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 缩略图
        pixmap = QPixmap(image_path).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label = QLabel(self)
        self.label.setPixmap(pixmap)
        layout.addWidget(self.label)

        # 删除按钮
        self.remove_button = QPushButton("删除", self)
        self.remove_button.clicked.connect(lambda: remove_callback(self))
        layout.addWidget(self.remove_button)

        self.setMinimumSize(QSize(220, 220))


########################################
# 加载 CLIP 模型
########################################
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

clip_threshold = 0.23  # CLIP 相似度阈值，视需求可修改


def get_clip_similarity(image_path, keyword):
    """
    使用 CLIP 模型计算图像和关键词的相似性
    """
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_tokens = clip.tokenize([keyword]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text_tokens)

    similarity = torch.cosine_similarity(image_features, text_features).item()
    return similarity


# ------------------------------------------------------
# 2) 视频播放器类（可进行播放、暂停、快进、设置剪辑起止点等）
#    这里稍作改动，增加一个 load_video(path) 方法方便在主窗口中调用
# ------------------------------------------------------
class VideoPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("视频播放器与剪辑工具")
        self.resize(800, 600)

        # 初始化播放器和视频窗口
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.error.connect(self.handleError)

        self.videoWidget = QVideoWidget()

        # 打开文件按钮（在本例中可不必经常使用，你也可以保留以便灵活打开其他视频）
        self.openButton = QPushButton("打开视频")
        self.openButton.clicked.connect(self.open_file)

        # 播放、暂停、快进按钮
        self.playButton = QPushButton("播放")
        self.playButton.clicked.connect(self.play_video)
        self.pauseButton = QPushButton("暂停")
        self.pauseButton.clicked.connect(self.pause_video)
        self.ffButton = QPushButton("快进10秒")
        self.ffButton.clicked.connect(self.fast_forward)

        # 进度条
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.set_position)

        # 标签显示当前时间和总时长
        self.timeLabel = QLabel("00:00 / 00:00")

        # 剪辑控制按钮：设置剪辑起点、终点及剪辑
        self.setStartButton = QPushButton("设置剪辑起点")
        self.setStartButton.clicked.connect(self.set_trim_start)
        self.setEndButton = QPushButton("设置剪辑终点")
        self.setEndButton.clicked.connect(self.set_trim_end)
        self.trimButton = QPushButton("剪辑视频")
        self.trimButton.clicked.connect(self.trim_video)

        # 初始化剪辑起点和终点（单位：毫秒）
        self.trim_start = None
        self.trim_end = None
        self.video_path = None  # 保存当前视频路径

        # 布局
        controlLayout = QHBoxLayout()
        controlLayout.addWidget(self.openButton)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.pauseButton)
        controlLayout.addWidget(self.ffButton)

        trimLayout = QHBoxLayout()
        trimLayout.addWidget(self.setStartButton)
        trimLayout.addWidget(self.setEndButton)
        trimLayout.addWidget(self.trimButton)

        layout = QVBoxLayout()
        layout.addWidget(self.videoWidget)
        layout.addWidget(self.positionSlider)
        layout.addWidget(self.timeLabel)
        layout.addLayout(controlLayout)
        layout.addLayout(trimLayout)

        self.setLayout(layout)

        # 设置视频输出
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)

    def open_file(self):
        """
        用于手动打开视频文件
        """
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", os.getcwd(),
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv *.mpeg *.mpg)"
        )
        if filename:
            self.load_video(filename)

    def load_video(self, path):
        """
        在主窗口中生成视频后，可直接调用此方法把视频路径传递给播放器
        """
        if not os.path.exists(path):
            QMessageBox.warning(self, "警告", f"文件不存在: {path}")
            return
        self.video_path = path
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
        self.playButton.setEnabled(True)
        # 重置剪辑起点与终点
        self.trim_start = None
        self.trim_end = None
        self.timeLabel.setText("00:00 / 00:00")

    def play_video(self):
        if self.mediaPlayer.mediaStatus() == QMediaPlayer.NoMedia:
            QMessageBox.warning(self, "警告", "请先打开视频文件！")
            return
        self.mediaPlayer.play()

    def pause_video(self):
        self.mediaPlayer.pause()

    def fast_forward(self):
        new_position = self.mediaPlayer.position() + 10000  # 快进10秒
        if new_position < self.mediaPlayer.duration():
            self.mediaPlayer.setPosition(new_position)
        else:
            self.mediaPlayer.setPosition(self.mediaPlayer.duration())

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    def position_changed(self, position):
        self.positionSlider.setValue(position)
        duration = self.mediaPlayer.duration()
        current_time = QTime(0, 0, 0).addMSecs(position).toString("mm:ss")
        total_time = QTime(0, 0, 0).addMSecs(duration).toString("mm:ss")
        self.timeLabel.setText(f"{current_time} / {total_time}")

    def duration_changed(self, duration):
        self.positionSlider.setRange(0, duration)

    def set_trim_start(self):
        self.trim_start = self.mediaPlayer.position()
        QMessageBox.information(self, "提示", f"剪辑起点已设置：{self.trim_start // 1000} 秒")

    def set_trim_end(self):
        self.trim_end = self.mediaPlayer.position()
        QMessageBox.information(self, "提示", f"剪辑终点已设置：{self.trim_end // 1000} 秒")

    def trim_video(self):
        """
        使用 ffmpeg 进行无损剪辑（-c copy），不重新编码
        """
        if not self.video_path:
            QMessageBox.warning(self, "警告", "请先打开视频文件！")
            return
        if self.trim_start is None or self.trim_end is None:
            QMessageBox.warning(self, "警告", "请先设置剪辑起点与终点！")
            return
        if self.trim_start >= self.trim_end:
            QMessageBox.warning(self, "警告", "剪辑起点必须小于剪辑终点！")
            return

        # 选择保存路径
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存剪辑后的视频", os.getcwd(),
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv *.mpeg *.mpg)"
        )
        if not save_path:
            return

        start_sec = self.trim_start / 1000
        end_sec = self.trim_end / 1000

        cmd = [
            "ffmpeg",
            "-y",                # 覆盖输出
            "-i", self.video_path,
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-c", "copy",
            save_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                QMessageBox.information(self, "成功", "视频剪辑成功！")
            else:
                error_msg = result.stderr
                QMessageBox.critical(self, "错误", f"视频剪辑失败：{error_msg}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"视频剪辑异常：{str(e)}")

    def handleError(self):
        error_string = self.mediaPlayer.errorString()
        QMessageBox.critical(self, "播放错误", f"错误信息：{error_string}")


# ------------------------------------------------------
# 3) 主界面：图像加载、匹配、生成视频，然后用上面的视频播放器来播放与剪辑
# ------------------------------------------------------
class ImageMatcherApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_paths = []      # 保存加载的图片路径
        self.image_widgets = []    # 保存图片对应的 QFrame
        self.matched_images = []   # 保存匹配的图片路径
        self.generated_video_path = None  # 保存生成的视频路径

    def initUI(self):
        self.setWindowTitle('Image Matcher App')
        self.resize(1000, 700)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 输入框：关键词
        self.keyword_input = QLineEdit(self)
        self.keyword_input.setPlaceholderText('输入关键词')
        main_layout.addWidget(self.keyword_input)

        # 加载图片按钮
        self.load_button = QPushButton('加载图片', self)
        self.load_button.clicked.connect(self.load_images)
        main_layout.addWidget(self.load_button)

        # 上方：滚动区域显示加载的图片
        self.loaded_images_scroll = QScrollArea(self)
        self.loaded_images_scroll.setWidgetResizable(True)
        self.loaded_images_container = QWidget()
        self.loaded_images_layout = QVBoxLayout(self.loaded_images_container)
        self.loaded_images_layout.setContentsMargins(10, 10, 10, 10)
        self.loaded_images_layout.setSpacing(10)
        self.loaded_images_scroll.setWidget(self.loaded_images_container)
        main_layout.addWidget(self.loaded_images_scroll)

        # 匹配图片按钮
        self.match_button = QPushButton('匹配图片', self)
        self.match_button.clicked.connect(self.match_images)
        main_layout.addWidget(self.match_button)

        # 中部：匹配结果列表
        self.match_list_widget = MatchedListWidget()
        self.match_list_widget.setMinimumHeight(250)
        self.matched_list_scroll = QScrollArea(self)
        self.matched_list_scroll.setWidgetResizable(True)
        self.matched_list_scroll.setWidget(self.match_list_widget)
        main_layout.addWidget(self.matched_list_scroll)

        # 生成视频按钮
        self.generate_video_button = QPushButton('生成视频', self)
        self.generate_video_button.clicked.connect(self.on_generate_video)
        main_layout.addWidget(self.generate_video_button)

        # 播放生成的视频按钮（用我们新的 VideoPlayer）
        self.play_button = QPushButton('播放生成的视频', self)
        self.play_button.clicked.connect(self.play_generated_video)
        main_layout.addWidget(self.play_button)

        self.setLayout(main_layout)

    ########################################
    # 加载图片
    ########################################
    def load_images(self):
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
            options=options
        )
        if file_paths:
            for image_path in file_paths:
                self.add_image(image_path)

    def add_image(self, image_path):
        """添加图片到滚动区域（上方）"""
        self.image_paths.append(image_path)

        frame = QFrame(self)
        frame_layout = QHBoxLayout(frame)
        frame_layout.setContentsMargins(10, 10, 10, 10)
        frame_layout.setSpacing(10)

        pixmap = QPixmap(image_path).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label = QLabel(self)
        label.setPixmap(pixmap)
        frame_layout.addWidget(label)

        remove_button = QPushButton('删除', self)
        remove_button.clicked.connect(lambda: self.remove_image(image_path, frame))
        frame_layout.addWidget(remove_button)

        self.image_widgets.append(frame)
        self.loaded_images_layout.addWidget(frame)

    def remove_image(self, image_path, frame):
        """从界面和列表中删除图片"""
        if image_path in self.image_paths:
            self.image_paths.remove(image_path)
            self.loaded_images_layout.removeWidget(frame)
            frame.deleteLater()
            self.image_widgets.remove(frame)

    ########################################
    # 匹配图片
    ########################################
    def match_images(self):
        keyword = self.keyword_input.text().strip()
        self.matched_images.clear()
        if not keyword:
            QMessageBox.warning(self, "警告", "请先输入关键词！")
            return

        for image_path in self.image_paths:
            similarity = get_clip_similarity(image_path, keyword)
            if similarity >= clip_threshold:
                self.matched_images.append(image_path)

        self.show_matched_images()

    def show_matched_images(self):
        self.match_list_widget.clear()
        for image_path in self.matched_images:
            item = QListWidgetItem()
            widget = MatchedItemWidget(image_path, self.remove_matched_item)
            item.setData(Qt.UserRole, widget)
            item.setSizeHint(widget.sizeHint())
            self.match_list_widget.addItem(item)
            self.match_list_widget.setItemWidget(item, widget)

    def remove_matched_item(self, widget):
        """
        widget: MatchedItemWidget 实例
        需要找到对应的 QListWidgetItem 并移除
        """
        for i in range(self.match_list_widget.count()):
            item = self.match_list_widget.item(i)
            stored_widget = item.data(Qt.UserRole)
            if stored_widget == widget:
                if widget.image_path in self.matched_images:
                    self.matched_images.remove(widget.image_path)
                self.match_list_widget.takeItem(i)
                break

    def get_ordered_matched_images(self):
        images = []
        for i in range(self.match_list_widget.count()):
            item = self.match_list_widget.item(i)
            widget = item.data(Qt.UserRole)
            if widget is not None:
                images.append(widget.image_path)
        return images

    ########################################
    # 生成视频（带淡入淡出）
    ########################################
    def on_generate_video(self):
        ordered_images = self.get_ordered_matched_images()
        self.save_video(ordered_images)

    def save_video(self, images):
        if not images:
            QMessageBox.warning(self, "警告", "没有匹配到任何图片，无法生成视频")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "保存视频", "", "视频文件 (*.mp4)")
        if not save_path:
            return

        frame_rate = 5
        duration_per_image = 2
        transition_duration = 1
        frames_per_image = int(frame_rate * duration_per_image)

        # 找一个合适的尺寸（取所有图片中最小的宽高）
        resolutions = [Image.open(img).size for img in images]
        target_size = (min(res[0] for res in resolutions), min(res[1] for res in resolutions))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_path, fourcc, frame_rate, target_size)

        def cross_fade(frame1, frame2, frame_rate, duration):
            frames = []
            total_frames = int(duration * frame_rate)
            for i in range(total_frames):
                alpha = i / total_frames
                blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
                frames.append(blended)
            return frames

        for idx, image_path in enumerate(images):
            with Image.open(image_path) as img:
                img = img.resize(target_size)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                # 固定时长显示
                for _ in range(frames_per_image):
                    video_writer.write(frame)

                # 与下一张图做淡入淡出
                if idx < len(images) - 1:
                    next_image_path = images[idx + 1]
                    with Image.open(next_image_path) as next_img:
                        next_img = next_img.resize(target_size)
                        next_frame = cv2.cvtColor(np.array(next_img), cv2.COLOR_RGB2BGR)
                        transition_frames = cross_fade(frame, next_frame, frame_rate, transition_duration)
                        for tf in transition_frames:
                            video_writer.write(tf)

        video_writer.release()
        self.generated_video_path = save_path
        QMessageBox.information(self, "完成", f"视频已保存到 {save_path}")

    ########################################
    # 播放生成的视频（使用 VideoPlayer）
    ########################################
    def play_generated_video(self):
        if not self.generated_video_path:
            QMessageBox.warning(self, "警告", "当前没有生成的视频可播放")
            return
        if not os.path.exists(self.generated_video_path):
            QMessageBox.warning(self, "错误", f"视频文件不存在：{self.generated_video_path}")
            return

        # 打开 VideoPlayer 窗口，并把生成视频的路径传给它
        self.video_player = VideoPlayer()
        self.video_player.load_video(self.generated_video_path)
        self.video_player.show()


# ------------------------------------------------------
# 4) 入口函数
# ------------------------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageMatcherApp()
    window.show()
    sys.exit(app.exec_())
