from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import requests
from langchain_core.tools import BaseTool
from PIL import Image
from pydantic import BaseModel, Field


class MediaParseInput(BaseModel):
    media_url: str = Field(description="媒体文件URL")
    media_type: str = Field(description="媒体类型： text/image/video")
    task_id: str = Field(description="审核任务ID")


class MediaParseTool(BaseTool):
    name: str = "media_parse_tool"
    description: str = "解析图文/视频媒体文件，返回预处理后的内容（文本/图片/视频关键帧）"
    args_schema = MediaParseInput
    return_direct = True

    def _run(self, media_url: str, media_type: str, task_id: str) -> dict:
        media_type = media_type.lower().strip()
        if media_type not in {"text", "image", "video"}:
            return {"status": "fail", "msg": f"不支持的媒体类型: {media_type}"}

        media_path = self._download_media(media_url, task_id, media_type)
        if not media_path:
            return {"status": "fail", "msg": "文件下载失败"}

        try:
            if media_type == "text":
                with open(media_path, "r", encoding="utf-8") as file:
                    content = file.read()
                return {
                    "status": "success",
                    "task_id": task_id,
                    "media_type": "text",
                    "media_path": media_path,
                    "content": content,
                }

            if media_type == "image":
                parsed = self._parse_image(media_path)
                return {
                    "status": "success",
                    "task_id": task_id,
                    "media_type": "image",
                    "media_path": parsed["image_path"],
                    "width": parsed["width"],
                    "height": parsed["height"],
                }

            frame_paths = self._extract_video_frames(media_path, task_id)
            return {
                "status": "success",
                "task_id": task_id,
                "media_type": "video",
                "media_path": media_path,
                "frame_paths": frame_paths,
                "frame_count": len(frame_paths),
            }
        except Exception as exc:  # noqa: BLE001
            return {"status": "fail", "task_id": task_id, "msg": f"媒体解析异常: {exc}"}

    def _download_media(self, media_url: str, task_id: str, media_type: str) -> str | None:
        suffix_map = {"text": ".txt", "image": ".jpg", "video": ".mp4"}
        suffix = suffix_map.get(media_type, ".bin")
        target_dir = Path(tempfile.gettempdir()) / "audit_engine" / task_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"source{suffix}"

        if media_url.startswith("http://") or media_url.startswith("https://"):
            resp = requests.get(media_url, timeout=30)
            if resp.status_code != 200:
                return None
            target_file.write_bytes(resp.content)
            return str(target_file)

        local_path = Path(media_url)
        if not local_path.exists():
            return None
        bytes_data = local_path.read_bytes()
        target_file.write_bytes(bytes_data)
        return str(target_file)

    def _parse_image(self, image_path: str) -> dict:
        image = Image.open(image_path)
        image.thumbnail((1024, 1024))
        image = image.convert("RGB")
        image.save(image_path, format="JPEG", quality=90)
        width, height = image.size
        return {"image_path": image_path, "width": width, "height": height}

    def _extract_video_frames(self, video_path: str, task_id: str, fps: int = 1, max_frames: int = 24) -> list[str]:
        output_dir = Path(tempfile.gettempdir()) / "audit_engine" / task_id / "frames"
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("视频文件无法打开")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(int(source_fps // fps), 1)
        current_frame = 0
        saved = 0
        frame_paths: list[str] = []

        while saved < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame % step == 0:
                frame_path = output_dir / f"frame_{saved:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))
                saved += 1
            current_frame += 1

        cap.release()
        return frame_paths