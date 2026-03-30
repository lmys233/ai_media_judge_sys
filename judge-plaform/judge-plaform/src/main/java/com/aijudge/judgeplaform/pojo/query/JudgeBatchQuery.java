package com.aijudge.judgeplaform.pojo.query;

import lombok.Data;

import java.util.List;

@Data
public class JudgeBatchQuery {
    private List<Long> caseIds;
}
