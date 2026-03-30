package com.aijudge.judgeplaform.pojo.mq;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AuditResultMessage {
    @Builder.Default
    private String schema_version = "1.0.0";
    private String case_id;
    private String trace_id;
    private String final_label;
    private Double confidence;
    private String status;
    private String reason;
    private String processed_at;
    private String source;
    private Map<String, Object> metadata;
    private List<ViolationDetail> violation_details;
}
