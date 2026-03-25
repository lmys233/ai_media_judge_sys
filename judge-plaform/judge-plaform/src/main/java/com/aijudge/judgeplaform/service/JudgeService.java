package com.aijudge.judgeplaform.service;

import com.aijudge.judgeplaform.pojo.query.JudgeSubmitQuery;
import org.springframework.web.multipart.MultipartFile;

public interface JudgeService {
    Long submit(JudgeSubmitQuery query, MultipartFile file);
}
