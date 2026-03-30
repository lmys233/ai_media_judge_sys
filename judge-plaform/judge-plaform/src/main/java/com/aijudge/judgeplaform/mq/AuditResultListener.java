package com.aijudge.judgeplaform.mq;

import com.aijudge.judgeplaform.mapper.JudgeCaseMapper;
import com.aijudge.judgeplaform.pojo.domain.JudgeCase;
import com.aijudge.judgeplaform.pojo.mq.AuditResultMessage;
import com.aijudge.judgeplaform.pojo.mq.ViolationDetail;
import com.aijudge.judgeplaform.sse.SseEmitterManager;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.List;

@Slf4j
@Component
@RequiredArgsConstructor
public class AuditResultListener {
    private final JudgeCaseMapper judgeCaseMapper;
    private final SseEmitterManager sseEmitterManager;
    private final ObjectMapper objectMapper = new ObjectMapper();

    private static final String QUEUE_NAME = "audit.result";

    @RabbitListener(queues = QUEUE_NAME)
    public void handleAuditResult(String messageJson) {
        log.info("Received audit result message: {}", messageJson);
        try {
            AuditResultMessage result = objectMapper.readValue(messageJson, AuditResultMessage.class);
            Long caseId = Long.parseLong(result.getCase_id());

            JudgeCase judgeCase = judgeCaseMapper.selectById(caseId);
            if (judgeCase == null) {
                log.error("Case not found: caseId={}", caseId);
                return;
            }

            // Update audit status
            judgeCase.setAuditStatus(result.getStatus());
            judgeCase.setAiResult(buildAiResultJson(result));
            judgeCase.setAiProcessedAt(LocalDateTime.now());
            // Mark as not viewed so user sees notification
            judgeCase.setIsViewed(false);

            // Update violation types and evidence from violation_details
            if (result.getViolation_details() != null && !result.getViolation_details().isEmpty()) {
                // Extract all violation types as JSON array string
                List<String> violationTypesList = result.getViolation_details().stream()
                        .map(ViolationDetail::getViolation_type)
                        .distinct()
                        .toList();
                judgeCase.setViolationTypes(objectMapper.writeValueAsString(violationTypesList));

                // Extract all evidence as JSON array string
                List<String> allEvidence = result.getViolation_details().stream()
                        .flatMap(vd -> vd.getEvidence() != null ? vd.getEvidence().stream() : java.util.stream.Stream.empty())
                        .distinct()
                        .toList();
                judgeCase.setEvidence(objectMapper.writeValueAsString(allEvidence));
            } else {
                // Fallback to final_label if no violation_details
                judgeCase.setViolationTypes(objectMapper.writeValueAsString(List.of(result.getFinal_label())));
                judgeCase.setEvidence("[]");
            }

            judgeCaseMapper.updateById(judgeCase);
            log.info("Updated case audit result: caseId={}, status={}, label={}, confidence={}, violation_types={}",
                    caseId, result.getStatus(), result.getFinal_label(), result.getConfidence(), judgeCase.getViolationTypes());

            // SSE推送通知给前端
            sendSseNotification(judgeCase, result);

        } catch (Exception e) {
            log.error("Failed to process audit result message: {}", messageJson, e);
        }
    }

    private String buildAiResultJson(AuditResultMessage result) {
        try {
            return objectMapper.writeValueAsString(new AiResult(
                    result.getFinal_label(),
                    result.getConfidence(),
                    result.getReason(),
                    result.getViolation_details()
            ));
        } catch (Exception e) {
            log.error("Failed to build ai result JSON", e);
            return "{}";
        }
    }

    private void sendSseNotification(JudgeCase judgeCase, AuditResultMessage result) {
        // 获取该案例的审核员ID
        Long reviewerId = judgeCase.getReviewerId();
        if (reviewerId == null) {
            log.warn("Case {} has no reviewerId, skip SSE notification", judgeCase.getCaseId());
            return;
        }

        // 构建简单推送数据（只发caseId，前端据此显示红点）
        SseNotification notification = new SseNotification(
                judgeCase.getCaseId(),
                result.getStatus(),
                result.getFinal_label()
        );

        sseEmitterManager.sendAuditResult(reviewerId, notification);
        log.info("SSE notification sent: reviewerId={}, caseId={}", reviewerId, judgeCase.getCaseId());
    }

    private record AiResult(String final_label, Double confidence, String reason, List<ViolationDetail> violation_details) {}

    // 简化的SSE通知，只包含必要信息
    private record SseNotification(Long caseId, String status, String finalLabel) {}
}
