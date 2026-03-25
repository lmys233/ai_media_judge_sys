package com.aijudge.judgeplaform.pojo.domain;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("judge_case")
public class JudgeCase {
    @TableId(value = "case_id", type = IdType.INPUT)
    private Long caseId;
    @TableField("media_id")
    private Long mediaId;
    @TableField("content_text")
    private String contentText;
    @TableField("media_type")
    private String mediaType;
    @TableField("violation_types")
    private String violationTypes;
    @TableField("review_reason")
    private String reviewReason;
    private String evidence;
    @TableField("media_url")
    private String mediaUrl;
    private String source;
    @TableField("reviewer_id")
    private Long reviewerId;
    @TableField("review_time")
    private java.time.LocalDateTime reviewTime;
}
