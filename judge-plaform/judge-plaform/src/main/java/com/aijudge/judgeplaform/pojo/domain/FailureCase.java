package com.aijudge.judgeplaform.pojo.domain;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("failure_case")
public class FailureCase {
    @TableId(value = "failure_id", type = IdType.INPUT)
    private Long failureId;

    @TableField("task_id")
    private String taskId;

    @TableField("trace_id")
    private String traceId;

    @TableField("biz_id")
    private Long bizId;

    @TableField("media_type")
    private String mediaType;

    @TableField("media_url")
    private String mediaUrl;

    @TableField("content_text")
    private String contentText;

    @TableField("error_message")
    private String errorMessage;

    @TableField("retry_count")
    private Integer retryCount;

    @TableField("failed_at")
    private LocalDateTime failedAt;

    @TableField("created_at")
    private LocalDateTime createdAt;
}
