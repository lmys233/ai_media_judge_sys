package com.aijudge.judgeplaform.pojo.mq;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FailureMessage {
    @Builder.Default
    private String schema_version = "1.0.0";
    private String task_id;
    private String trace_id;
    private Long biz_id;
    private String media_type;
    private String media_url;
    private String content_text;
    private String error_message;
    private String failed_at;
}
