package com.aijudge.judgeplaform.service;

import com.aijudge.judgeplaform.pojo.domain.JudgeCase;
import com.aijudge.judgeplaform.pojo.query.JudgeSubmitQuery;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

public interface JudgeService {
    Long submit(JudgeSubmitQuery query, MultipartFile file);

    JudgeCase getCaseById(Long caseId);

    List<JudgeCase> getUnviewedCases(Long reviewerId);

    void markCasesAsViewed(List<Long> caseIds);

    List<JudgeCase> getCasesByIds(List<Long> caseIds);
}
