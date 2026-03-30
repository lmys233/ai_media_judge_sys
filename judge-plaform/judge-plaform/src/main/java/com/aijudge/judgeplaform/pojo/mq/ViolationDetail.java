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
public class ViolationDetail {
    private String violation_type;
    private Double confidence;
    private List<String> evidence;
    private String reason;
}
