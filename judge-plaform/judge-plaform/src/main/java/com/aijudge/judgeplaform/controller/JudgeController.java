package com.aijudge.judgeplaform.controller;

import com.aijudge.judgeplaform.context.ReviewerContext;
import com.aijudge.judgeplaform.pojo.domain.JudgeCase;
import com.aijudge.judgeplaform.pojo.domain.R;
import com.aijudge.judgeplaform.pojo.query.JudgeBatchQuery;
import com.aijudge.judgeplaform.pojo.query.JudgeSubmitQuery;
import com.aijudge.judgeplaform.service.JudgeService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.stream.Collectors;

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

    /**
     * 根据caseId查询案例详情
     */
    @GetMapping("/get")
    public R<JudgeCase> getCaseById(@RequestParam("caseId") Long caseId) {
        try {
            JudgeCase judgeCase = judgeService.getCaseById(caseId);
            if (judgeCase == null) {
                return R.fail(404, "案例不存在");
            }
            return R.ok(judgeCase);
        } catch (Exception e) {
            return R.fail("查询失败: " + e.getMessage());
        }
    }

    /**
     * 获取未查看的审核通知列表
     * 调用此接口后，列表中的案例会被标记为已查看
     */
    @GetMapping("/unviewed")
    public R<List<JudgeCase>> getUnviewedCases() {
        try {
            Long reviewerId = ReviewerContext.getReviewerId();
            if (reviewerId == null) {
                return R.fail(401, "审核员ID不存在");
            }
            List<JudgeCase> cases = judgeService.getUnviewedCases(reviewerId);
            // Mark as viewed after fetching
            if (!cases.isEmpty()) {
                List<Long> caseIds = cases.stream()
                        .map(JudgeCase::getCaseId)
                        .collect(Collectors.toList());
                judgeService.markCasesAsViewed(caseIds);
            }
            return R.ok(cases);
        } catch (Exception e) {
            return R.fail("查询失败: " + e.getMessage());
        }
    }

    /**
     * 批量查询案例详情（用于AI引擎召回后获取审核理由等完整信息）
     */
    @PostMapping("/batch")
    public R<List<JudgeCase>> getCasesByIds(@RequestBody JudgeBatchQuery query) {
        try {
            if (query == null || query.getCaseIds() == null || query.getCaseIds().isEmpty()) {
                return R.ok(List.of());
            }
            // Limit batch size to avoid overload
            List<Long> limitedIds = query.getCaseIds().stream().limit(20).collect(Collectors.toList());
            List<JudgeCase> cases = judgeService.getCasesByIds(limitedIds);
            return R.ok(cases);
        } catch (Exception e) {
            return R.fail("批量查询失败: " + e.getMessage());
        }
    }
}
