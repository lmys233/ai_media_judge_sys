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
}
