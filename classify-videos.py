import os
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from sklearn.cluster import KMeans
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from shutil import move, copy


# 提取视频的帧作为特征
def extract_video_features(video_path, model, frame_interval=30):
    video = VideoFileClip(video_path)
    frame_count = 0
    features = []

    for frame in video.iter_frames(fps=1):  # 以每秒一帧的速度获取帧
        if frame_count % frame_interval == 0:
            # Resize frame to match model input size (299x299 for InceptionV3)
            img = cv2.resize(frame, (299, 299))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            # 提取特征
            feature = model.predict(img)
            features.append(feature.flatten())

        frame_count += 1

    # 取视频的所有帧特征的均值作为视频的最终特征
    return np.mean(features, axis=0)


# 批量提取目录下所有视频的特征
def extract_features_for_all_videos(input_dir, model, frame_interval=30):
    video_features = []
    video_files = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):  # 你可以根据需要修改文件格式
            video_path = os.path.join(input_dir, filename)
            print(f"正在处理视频: {filename}")
            features = extract_video_features(video_path, model, frame_interval)
            video_features.append(features)
            video_files.append(filename)

    return np.array(video_features), video_files


# 对视频进行聚类
def cluster_videos(video_features, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(video_features)
    return kmeans.labels_


# 将视频分类到不同的文件夹
def classify_videos(input_dir, output_dir, video_files, labels):
    for label, filename in zip(labels, video_files):
        output_folder = os.path.join(output_dir, f"cluster_{label}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_folder, filename)

        # move(input_path, output_path)
        copy(input_path, output_path)
        print(f"已将视频 {filename} 移动到 {output_folder}")


# 主函数
def main(input_dir, output_dir, num_clusters=3, frame_interval=30):
    """
    视频分类
    :param input_dir: 视频输入目录
    :param output_dir: 视频输出目录
    :param num_clusters: 确定要将数据分成多少个簇，影响分类数量
    :param frame_interval: 帧采样频率，默认每隔 30 帧（即每秒）提取一帧图像用于特征提取
    :return:
    """
    # 加载预训练的InceptionV3模型，并去掉顶层的分类部分，只用来提取特征
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    # 提取所有视频的特征
    video_features, video_files = extract_features_for_all_videos(input_dir, model, frame_interval)

    # 对视频进行聚类
    labels = cluster_videos(video_features, num_clusters)

    # 将视频移动到相应的分类文件夹
    classify_videos(input_dir, output_dir, video_files, labels)


# 示例调用
input_directory = "path/to/input_videos"
output_directory = "path/to/output_videos"
main(input_directory, output_directory, num_clusters=30, frame_interval=30)
