import copy
import os
import torch
import cv2
import numpy as np
import onnxruntime as ort

from moviepy.editor import VideoFileClip


class TransNetV2_Onnx():
    def __init__(self, onnx_path, window_size=100, overlap_size=20):
        self.onnx_path = onnx_path

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f'{onnx_path} is not found')

        self.ort_session = ort.InferenceSession(onnx_path)
        if self.ort_session is None:
            raise Exception(f'self.ort_session is None')

        self.window_size = window_size
        self.overlap_size = overlap_size

    def inference_video(self, input_video_path, output_video_dir, save=False):
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f'{input_video_path} is not found')

        print(f'-----------TransNetV2/inference_video----------')
        print(f'transnetv2 inference_video: {input_video_path}')

        # 打开视频文件
        cap = cv2.VideoCapture(input_video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f'video frame count:{frame_count}')

        all_predictions = []
        window_size = self.window_size  # 100
        step_size = 50  # 固定步长为50
        overlap_frame_size = 25

        # 先读取第一帧用于填充
        ret, first_frame = cap.read()
        if not ret:
            raise Exception("Cannot read first frame")
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        first_frame = cv2.resize(first_frame, (48, 27))

        # 重置视频读取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        current_frame = 0
        while current_frame < frame_count:
            # 构建100帧的窗口
            frames = []

            # 如果是开始部分，需要用第一帧填充
            if current_frame == 0:
                frames.extend([first_frame] * overlap_frame_size)

            # 读取实际帧
            actual_frames_to_read = min(window_size - len(frames), frame_count - current_frame)
            for _ in range(actual_frames_to_read):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (48, 27))
                    frames.append(frame)
                else:
                    break

            actual_frames = len(frames)

            # 如果不足100帧，用最后一帧填充
            if actual_frames < window_size:
                last_frame = frames[-1]
                padding_size = window_size - actual_frames
                frames.extend([last_frame] * padding_size)

            # 执行推理
            input_video = np.array(frames, dtype=np.uint8)
            input_video = np.expand_dims(input_video, axis=0)
            single_frame_pred, _ = self.ort_session.run(None, {'input': input_video})
            single_frame_pred = 1 / (1 + np.exp(-single_frame_pred))

            # 保存中间50帧的预测结果（第25帧到第74帧）
            predictions_to_save = single_frame_pred[0, overlap_frame_size:overlap_frame_size + step_size, 0]
            # 如果是最后一个窗口，可能需要截断
            if current_frame + step_size > frame_count:
                remaining_frames = frame_count - current_frame
                predictions_to_save = predictions_to_save[:remaining_frames]

            all_predictions.append(predictions_to_save)

            print(f"\r处理视频帧 {min(len(all_predictions) * 50, frame_count)}/{frame_count}", end="")

            # 移动到下一个窗口的起始位置
            current_frame += step_size
            if current_frame < frame_count:
                # 回退25帧以保持连续性
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - overlap_frame_size)

        print("")
        cap.release()

        # 合并所有预测结果
        single_frame_pred = np.concatenate(all_predictions)
        assert len(
            single_frame_pred) == frame_count, f"Predictions count {len(single_frame_pred)} doesn't match frame count {frame_count}"

        # 推理场景
        scenes = predictions_to_scenes(single_frame_pred)
        print(f'divide scene result: {scenes}')

        if save:
            if not os.path.exists(output_video_dir):
                os.makedirs(output_video_dir)

            input_video_name = os.path.basename(input_video_path).split('.')[0]
            output_txt_path = os.path.join(output_video_dir, f'{input_video_name}_scenes.txt')
            save_scenes_results_to_txt(scenes, output_txt_path)

            # 可视化视频
            visualize_video(input_video_path, scenes, output_video_dir)

        print(f'process completed')

        return scenes


def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
    predictions = (predictions > threshold).astype(np.uint8)

    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    # just fix if all predictions are 1
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)


def save_scenes_results_to_txt(scenes, output_txt_path):
    with open(output_txt_path, 'w') as f:
        for i, scene_index in enumerate(scenes):
            start_frame = scene_index[0]
            end_frame = scene_index[1]
            f.write(f'{start_frame} {end_frame}\n')

    print(f'save scenes results to {output_txt_path}')


def convert_to_h264(input_video_path):
    if not os.path.exists(input_video_path):
        raise Exception(f'{input_video_path} is not exist')

    # 在output_video_path修改名称，文件名加上moviepy_前缀
    new_output_video_path = input_video_path.replace('.mp4', '_h264.mp4')

    # Load the input video
    video_clip = VideoFileClip(input_video_path)

    # Write the output video with H.264 encoding
    video_clip.write_videofile(new_output_video_path, codec='libx264')


def visualize_video(input_video_path, scenes, output_video_dir):
    if not os.path.exists(input_video_path):
        raise Exception(f'{input_video_path} is not exist')

    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)

    input_video_name = os.path.basename(input_video_path).split('.')[0]
    output_video_path = os.path.join(output_video_dir, f'{input_video_name}_camera_detect.mp4')

    video_read_cap = cv2.VideoCapture(input_video_path)

    frame_video_width = int(video_read_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_video_height = int(video_read_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(video_read_cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(video_read_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_write_cap = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), frame_fps,
                                      (frame_video_width, frame_video_height))

    for i, scene_index in enumerate(scenes):
        start_frame = scene_index[0]
        end_frame = scene_index[1]
        num_frames = end_frame - start_frame

        scene_str = f'scene {i}'

        # 设置视频读取的起始位置
        video_read_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(num_frames):
            ret, frame = video_read_cap.read()
            if ret:
                cv2.putText(frame, scene_str, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 1, cv2.LINE_AA)
                video_write_cap.write(frame)
            else:
                break

    video_read_cap.release()
    video_write_cap.release()

    print(f'video: {output_video_dir} write success')

    # 使用moviepy将视频转换为H.264编码
    convert_to_h264(output_video_path)


if __name__ == '__main__':
    onnx_path = "./transnetv2.onnx"
    input_video_path = './inference_test/input_videos/input_video.mp4'
    output_video_dir = './inference_test/output_videos_onnx'

    transnetv2_onnx = TransNetV2_Onnx(onnx_path, window_size=100, overlap_size=20)
    transnetv2_onnx.inference_video(input_video_path, output_video_dir, save_scene_txt=True)