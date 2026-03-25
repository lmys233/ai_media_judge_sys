package com.aijudge.judgeplaform.pojo.mq;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ManualCaseMessage {
    @Builder.Default
    private String schema_version = "1.0.0";
    private String trace_id;
    private String case_id;
    private String media_type;
    private String content_text;
    private List<String> violation_types;
    private String review_reason;
    private List<String> evidence;
    private String source;
}
