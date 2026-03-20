from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class MediaParseInput(BaseModel):
    media_url : str = Field(description = "媒体文件URL")
    media_type : str = Field(description = "媒体类型： text/image/video")
    task_id: str = Field(description = "审核任务ID")

class MediaParseTool(BaseTool):
    name = "media_parse_tool"
    description = "解析二图文/视频媒体文件，返回预处理后的内容（文本/图片/视频关键帧"
    args_schema = MediaParseInput
    return_direct = True

    def _run(self, media_url: str, media_type: type, task_id: str):
        # 下载媒体文件
        media_path = self._download_media(media_url, task_id)
        if not media_path:
            return {"status": "fail", "msg": "文件下载失败"}