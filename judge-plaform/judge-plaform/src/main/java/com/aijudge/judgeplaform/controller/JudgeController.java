package com.aijudge.judgeplaform.controller;

import com.aijudge.judgeplaform.pojo.domain.R;
import com.aijudge.judgeplaform.pojo.query.JudgeSubmitQuery;
import com.aijudge.judgeplaform.service.JudgeService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/judge")
@RequiredArgsConstructor
public class JudgeController {
    private final JudgeService judgeService;

    /**
     * 审核提交接口
     */
    @PostMapping("/submit")
    public R<String> submit(
            @RequestPart("data") JudgeSubmitQuery query,
            @RequestPart(value = "file", required = false) MultipartFile file) {
        try {
            Long caseId = judgeService.submit(query, file);
            return R.ok("提交成功，caseId=" + caseId);
        } catch (IllegalArgumentException e) {
            return R.fail(400, e.getMessage());
        } catch (Exception e) {
            return R.fail("提交失败: " + e.getMessage());
        }
    }
}
