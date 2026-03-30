package com.aijudge.judgeplaform.pojo.query;

import lombok.Data;

import java.util.List;

@Data
public class JudgeSubmitQuery {
    private String mediaType;
    private Long mediaId;
    private String content;
    private List<String> violationTypes;
    private String reviewReason;
    private List<String> evidence;
    /**
     * 审核模式: ai=AI自动审核, human=人工审核(默认)
     */
    private String auditMode = "human";
}
