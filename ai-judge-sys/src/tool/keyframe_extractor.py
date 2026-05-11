"""关键帧提取器模块

两阶段关键帧提取算法：
1. 阶段一：场景检测（帧差分算法）— 检测镜头边界，确定 N 个场景
2. 阶段二：每场景选代表性帧（信息量评分）— 每场景取 TOP 1-2 帧

输出：关键帧图片路径列表，可直接复用图片审核链路。
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Keyframe(NamedTuple):
    """关键帧"""
    frame_index: int  # 帧序号
    timestamp: float  # 时间戳（秒）
    path: str  # 保存路径
    info_score: float  # 信息量评分


class KeyframeExtractor:
    """关键帧提取器

    使用帧差分做场景检测，然后用信息量评分（梯度幅度）在每场景中选择代表性帧。
    """

    def __init__(
        self,
        max_keyframes: int = 16,
        fps: int = 1,
        min_scene_frames: int = 5,
        diff_threshold: float = 30.0,
    ) -> None:
        """初始化关键帧提取器

        Args:
            max_keyframes: 最多提取的关键帧数量
            fps: 帧差分时的采样帧率（每秒采样帧数）
            min_scene_frames: 场景最少帧数（小于此值则合并到相邻场景）
            diff_threshold: 帧差分阈值（像素差异超此值认为场景切换）
        """
        self.max_keyframes = max_keyframes
        self.fps = fps
        self.min_scene_frames = min_scene_frames
        self.diff_threshold = diff_threshold

    def extract_keyframes(self, video_path: str, task_id: str | None = None) -> list[str]:
        """提取视频的关键帧

        Args:
            video_path: 视频文件路径
            task_id: 任务ID（用于临时文件目录）

        Returns:
            关键帧图片路径列表
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error("视频无法打开: %s", video_path)
                return []

            # 阶段一：检测场景边界
            shot_boundaries = self._detect_shot_boundaries(cap)
            cap.release()

            # 重新打开以提取帧
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []

            # 阶段二：每场景选代表性帧
            keyframes = self._select_representative_frames(cap, shot_boundaries, task_id)
            cap.release()

            logger.info(
                "关键帧提取完成: video=%s, scenes=%d, keyframes=%d",
                video_path,
                len(shot_boundaries) + 1,
                len(keyframes),
            )
            return [kf.path for kf in keyframes]

        except Exception as e:
            logger.error("关键帧提取异常: video=%s, error=%s", video_path, e)
            return []

    def _detect_shot_boundaries(self, cap: cv2.VideoCapture) -> list[int]:
        """检测场景边界（镜头切换点）

        使用帧差分算法：计算相邻帧的像素差异，超过阈值则认为是场景切换。

        Args:
            cap: 打开的视频捕获器

        Returns:
            场景边界帧索引列表（如 [30, 90, 150] 表示第30、90、150帧是场景切换点）
        """
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(int(source_fps / self.fps), 1)

        boundaries: list[int] = []
        prev_gray: np.ndarray | None = None
        current_frame = 0
        sampled_index = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # 计算帧差分
                diff = cv2.absdiff(prev_gray, gray)
                mean_diff = np.mean(diff)

                if mean_diff > self.diff_threshold:
                    # 场景切换，标记边界
                    boundaries.append(sampled_index)

            prev_gray = gray
            current_frame += step
            sampled_index += 1

            if current_frame >= total_frames:
                break

        # 合并过短的场景
        boundaries = self._merge_short_scenes(boundaries, sampled_index)

        logger.debug("场景边界检测完成: total_sampled=%d, boundaries=%s", sampled_index, boundaries)
        return boundaries

    def _merge_short_scenes(self, boundaries: list[int], total_frames: int) -> list[int]:
        """合并过短的场景

        如果两个场景边界之间的帧数小于 min_scene_frames，则合并到相邻场景。
        """
        if not boundaries:
            return []

        merged: list[int] = []
        prev_boundary = 0

        for boundary in boundaries:
            scene_length = boundary - prev_boundary
            if scene_length >= self.min_scene_frames:
                merged.append(boundary)
                prev_boundary = boundary
            # else: 忽略这个边界，将其合并到前一个场景

        return merged

    def _compute_info_score(self, frame: np.ndarray) -> float:
        """计算帧的信息量评分

        使用梯度幅度作为信息量指标：边缘和纹理丰富的帧信息量更高。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Sobel 梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        # 返回梯度幅度均值作为评分
        return float(np.mean(magnitude))

    def _select_representative_frames(
        self,
        cap: cv2.VideoCapture,
        shot_boundaries: list[int],
        task_id: str | None,
    ) -> list[Keyframe]:
        """在每个场景中选择代表性帧

        Args:
            cap: 打开的视频捕获器
            shot_boundaries: 场景边界帧索引列表
            task_id: 任务ID（用于临时文件目录）

        Returns:
            关键帧列表
        """
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(int(source_fps / self.fps), 1)

        # 构建场景列表：每个场景的起始和结束帧索引
        scenes: list[tuple[int, int]] = []
        prev_start = 0
        for boundary in sorted(shot_boundaries):
            scenes.append((prev_start, boundary))
            prev_start = boundary
        # 最后一个场景
        scenes.append((prev_start, total_frames // step))

        # 为每个场景选择代表性帧
        keyframes: list[Keyframe] = []
        frames_per_scene = max(1, self.max_keyframes // max(len(scenes), 1))

        for scene_idx, (start, end) in enumerate(scenes):
            scene_keyframes = self._find_top_k_frames_in_scene(cap, start, end, frames_per_scene, source_fps, task_id)
            keyframes.extend(scene_keyframes)

            if len(keyframes) >= self.max_keyframes:
                # 达到上限，按信息量排序取 TOP
                keyframes = sorted(keyframes, key=lambda k: k.info_score, reverse=True)[:self.max_keyframes]
                break

        # 按时间顺序排序
        keyframes = sorted(keyframes, key=lambda k: k.frame_index)

        logger.debug("代表性帧选择完成: scenes=%d, keyframes=%d", len(scenes), len(keyframes))
        return keyframes

    def _find_top_k_frames_in_scene(
        self,
        cap: cv2.VideoCapture,
        start: int,
        end: int,
        k: int,
        source_fps: float,
        task_id: str | None,
    ) -> list[Keyframe]:
        """在场景[start, end)范围内找信息量最高的 k 帧

        Args:
            cap: 打开的视频捕获器
            start: 起始帧索引
            end: 结束帧索引
            k: 需要选择的帧数
            source_fps: 原始视频帧率

        Returns:
            关键帧列表
        """
        if start >= end:
            return []

        # 采样该场景内的所有帧
        candidates: list[tuple[int, float, np.ndarray]] = []  # (frame_index, info_score, frame)

        for sampled_idx in range(start, min(end, start + 30)):  # 最多采样30个候选帧
            frame_pos = sampled_idx * max(int(source_fps / self.fps), 1)
            if frame_pos >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            info_score = self._compute_info_score(frame)
            candidates.append((sampled_idx, info_score, frame.copy()))

        if not candidates:
            return []

        # 取信息量最高的 k 帧
        top_k = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]

        # 构建 Keyframe 对象并保存
        result: list[Keyframe] = []
        for frame_idx, info_score, frame in top_k:
            # 保存到临时文件
            if task_id:
                output_dir = Path(tempfile.gettempdir()) / "audit_engine" / task_id / "keyframes"
            else:
                output_dir = Path(tempfile.gettempdir()) / "audit_engine" / "keyframes"
            output_dir.mkdir(parents=True, exist_ok=True)

            frame_path = output_dir / f"kf_{frame_idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)

            timestamp = frame_idx * max(int(source_fps / self.fps), 1) / source_fps
            result.append(Keyframe(
                frame_index=frame_idx,
                timestamp=timestamp,
                path=str(frame_path),
                info_score=info_score,
            ))

        return result


def extract_keyframes(video_path: str, task_id: str | None = None, max_keyframes: int = 16) -> list[str]:
    """便捷函数：提取视频关键帧

    Args:
        video_path: 视频文件路径
        task_id: 任务ID
        max_keyframes: 最多提取的关键帧数量

    Returns:
        关键帧图片路径列表
    """
    extractor = KeyframeExtractor(max_keyframes=max_keyframes)
    return extractor.extract_keyframes(video_path, task_id)
