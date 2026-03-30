package com.aijudge.judgeplaform.service.impl;

import com.aijudge.judgeplaform.context.ReviewerContext;
import com.aijudge.judgeplaform.mapper.JudgeCaseMapper;
import com.aijudge.judgeplaform.pojo.domain.JudgeCase;
import com.aijudge.judgeplaform.pojo.mq.ManualCaseMessage;
import com.aijudge.judgeplaform.pojo.query.JudgeSubmitQuery;
import com.aijudge.judgeplaform.service.FileService;
import com.aijudge.judgeplaform.service.FileService;
import com.aijudge.judgeplaform.service.JudgeService;
import com.aijudge.judgeplaform.support.SnowflakeIdGenerator;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import lombok.RequiredArgsConstructor;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.MultipartFile;

import java.time.LocalDateTime;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class JudgeServiceImpl implements JudgeService {
    private static final int MAX_REASON_LEN = 50;
    private static final int MAX_EVIDENCE_LEN = 50;
    private static final Set<String> ALLOWED_VIOLATION_TYPES = Set.of(
            "normal", "abuse", "violence", "porn", "politics", "other"
    );
    private static final Set<String> ALLOWED_MEDIA_TYPES = Set.of("text", "image", "video");
    private static final String AI_TASK_TOPIC = "audit.task";
    private static final String MANUAL_CASE_TOPIC = "audit.manual.case";
    private static final String AUDIT_MODE_AI = "ai";
    private static final String AUDIT_MODE_HUMAN = "human";

    private final JudgeCaseMapper judgeCaseMapper;
    private final RabbitTemplate rabbitTemplate;
    private final SnowflakeIdGenerator snowflakeIdGenerator;
    private final FileService fileService;

    @Override
    public Long submit(JudgeSubmitQuery query, MultipartFile file) {
        validate(query, file);
        Long reviewerId = ReviewerContext.getReviewerId();
        if (reviewerId == null) {
            throw new IllegalArgumentException("reviewer ID not found in context");
        }

        String mediaType = query.getMediaType();
        String mediaUrl = null;
        if (!"text".equals(mediaType)) {
            mediaUrl = fileService.upload(file);
        }

        Long caseId = snowflakeIdGenerator.nextId();
        LocalDateTime reviewTime = LocalDateTime.now();

        JudgeCase judgeCase = new JudgeCase();
        judgeCase.setCaseId(caseId);
        judgeCase.setMediaId(query.getMediaId());
        judgeCase.setMediaType(mediaType);
        judgeCase.setViolationTypes(toJsonArray(query.getViolationTypes()));
        judgeCase.setReviewReason(query.getReviewReason());
        judgeCase.setEvidence(toJsonArray(query.getEvidence()));
        judgeCase.setSource("human");
        judgeCase.setReviewerId(reviewerId);
        judgeCase.setReviewTime(reviewTime);

        if ("text".equals(mediaType)) {
            judgeCase.setContentText(query.getContent());
            judgeCase.setMediaUrl(null);
        } else {
            judgeCase.setContentText(null);
            judgeCase.setMediaUrl(mediaUrl);
        }

        // Determine audit mode, default to human
        String auditMode = query.getAuditMode();
        if (auditMode == null) {
            auditMode = AUDIT_MODE_HUMAN;
        }
        auditMode = auditMode.trim().toLowerCase();
        boolean isAiMode = AUDIT_MODE_AI.equals(auditMode);

        // Set initial audit status
        judgeCase.setAuditStatus(isAiMode ? "pending" : "pending");
        judgeCase.setRetryCount(0);
        judgeCase.setSource(isAiMode ? "ai" : "human");
        // Human submissions are already "viewed" since user just submitted them
        judgeCase.setIsViewed(true);

        judgeCaseMapper.insert(judgeCase);

        String traceId = UUID.randomUUID().toString();
        String contentForMq = "text".equals(mediaType) ? query.getContent() : mediaUrl;

        if (isAiMode) {
            // AI自动审核：发送到audit.task队列
            String now = LocalDateTime.now().toString();
            String aiTaskJson = "{"
                    + "\"schema_version\":\"1.0.0\","
                    + "\"trace_id\":\"" + escapeJson(traceId) + "\","
                    + "\"task_id\":\"" + escapeJson(String.valueOf(caseId)) + "\","
                    + "\"biz_id\":\"" + escapeJson(query.getMediaId() != null ? String.valueOf(query.getMediaId()) : "") + "\","
                    + "\"media_type\":\"" + escapeJson(mediaType) + "\","
                    + "\"media_url\":\"" + escapeJson(mediaUrl != null ? mediaUrl : "") + "\","
                    + "\"content_text\":\"" + escapeJson(contentForMq) + "\","
                    + "\"lang\":\"zh\","
                    + "\"priority\":5,"
                    + "\"source\":\"ai_submission\","
                    + "\"created_at\":\"" + escapeJson(now) + "\","
                    + "\"retry_count\":0,"
                    + "\"max_retry\":3,"
                    + "\"dedupe_key\":\"" + escapeJson(String.valueOf(caseId)) + "\""
                    + "}";
            rabbitTemplate.convertAndSend(AI_TASK_TOPIC, aiTaskJson);
        } else {
            // 人工审核：发送到audit.manual.case队列
            ManualCaseMessage message = ManualCaseMessage.builder()
                    .trace_id(traceId)
                    .case_id(String.valueOf(caseId))
                    .media_type(mediaType)
                    .content_text(contentForMq)
                    .violation_types(query.getViolationTypes())
                    .review_reason(query.getReviewReason())
                    .evidence(query.getEvidence())
                    .source("human")
                    .build();
            rabbitTemplate.convertAndSend(MANUAL_CASE_TOPIC, toMessageJson(message));
        }

        return caseId;
    }

    private void validate(JudgeSubmitQuery query, MultipartFile file) {
        if (query == null) {
            throw new IllegalArgumentException("request body is empty");
        }
        String mediaType = query.getMediaType();
        if (!StringUtils.hasText(mediaType)) {
            mediaType = "text";
            query.setMediaType(mediaType);
        }
        mediaType = mediaType.trim().toLowerCase();
        query.setMediaType(mediaType);
        if (!ALLOWED_MEDIA_TYPES.contains(mediaType)) {
            throw new IllegalArgumentException("invalid mediaType: " + mediaType);
        }

        if ("text".equals(mediaType)) {
            if (!StringUtils.hasText(query.getContent())) {
                throw new IllegalArgumentException("content is required when mediaType=text");
            }
        } else {
            if (file == null || file.isEmpty()) {
                throw new IllegalArgumentException("file is required when mediaType=" + mediaType);
            }
        }

        if (!StringUtils.hasText(query.getReviewReason())) {
            throw new IllegalArgumentException("reviewReason is required");
        }
        if (query.getReviewReason().length() > MAX_REASON_LEN) {
            throw new IllegalArgumentException("reviewReason max " + MAX_REASON_LEN + " chars");
        }
        if (CollectionUtils.isEmpty(query.getViolationTypes())) {
            throw new IllegalArgumentException("violationTypes is required");
        }
        LinkedHashSet<String> normalizedTypeSet = new LinkedHashSet<>();
        for (String type : query.getViolationTypes()) {
            if (!StringUtils.hasText(type)) {
                throw new IllegalArgumentException("violationTypes has blank value");
            }
            String normalized = type.trim().toLowerCase();
            if (!ALLOWED_VIOLATION_TYPES.contains(normalized)) {
                throw new IllegalArgumentException("invalid violation type: " + type);
            }
            normalizedTypeSet.add(normalized);
        }
        query.setViolationTypes(List.copyOf(normalizedTypeSet));

        boolean normalOnly = query.getViolationTypes().size() == 1
                && "normal".equals(query.getViolationTypes().get(0));

        if (CollectionUtils.isEmpty(query.getEvidence())) {
            if (!normalOnly) {
                throw new IllegalArgumentException("evidence is required when violation type is not normal");
            }
            query.setEvidence(List.of());
        } else {
            LinkedHashSet<String> normalizedEvidenceSet = new LinkedHashSet<>();
            for (String ev : query.getEvidence()) {
                if (!StringUtils.hasText(ev)) {
                    throw new IllegalArgumentException("evidence has blank value");
                }
                String normalizedEv = ev.trim();
                if (normalizedEv.length() > MAX_EVIDENCE_LEN) {
                    throw new IllegalArgumentException("evidence item max " + MAX_EVIDENCE_LEN + " chars");
                }
                normalizedEvidenceSet.add(normalizedEv);
            }
            query.setEvidence(List.copyOf(normalizedEvidenceSet));
        }
    }

    private String toJsonArray(List<String> items) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) {
                sb.append(",");
            }
            sb.append("\"").append(escapeJson(items.get(i))).append("\"");
        }
        sb.append("]");
        return sb.toString();
    }

    private String toMessageJson(ManualCaseMessage message) {
        return "{"
                + "\"schema_version\":\"" + escapeJson(message.getSchema_version()) + "\","
                + "\"trace_id\":\"" + escapeJson(message.getTrace_id()) + "\","
                + "\"case_id\":\"" + escapeJson(message.getCase_id()) + "\","
                + "\"media_type\":\"" + escapeJson(message.getMedia_type()) + "\","
                + "\"content_text\":\"" + escapeJson(message.getContent_text()) + "\","
                + "\"violation_types\":" + toJsonArray(message.getViolation_types()) + ","
                + "\"review_reason\":\"" + escapeJson(message.getReview_reason()) + "\","
                + "\"evidence\":" + toJsonArray(message.getEvidence()) + ","
                + "\"source\":\"" + escapeJson(message.getSource()) + "\""
                + "}";
    }

    private String escapeJson(String value) {
        if (value == null) {
            return "";
        }
        return value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\b", "\\b")
                .replace("\f", "\\f")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    @Override
    public JudgeCase getCaseById(Long caseId) {
        if (caseId == null) {
            return null;
        }
        return judgeCaseMapper.selectById(caseId);
    }

    @Override
    public List<JudgeCase> getUnviewedCases(Long reviewerId) {
        if (reviewerId == null) {
            return List.of();
        }
        LambdaQueryWrapper<JudgeCase> wrapper = Wrappers.lambdaQuery();
        wrapper.eq(JudgeCase::getReviewerId, reviewerId)
                .eq(JudgeCase::getIsViewed, false)
                .orderByDesc(JudgeCase::getAiProcessedAt)
                .orderByDesc(JudgeCase::getReviewTime);
        return judgeCaseMapper.selectList(wrapper);
    }

    @Override
    public void markCasesAsViewed(List<Long> caseIds) {
        if (caseIds == null || caseIds.isEmpty()) {
            return;
        }
        for (Long caseId : caseIds) {
            JudgeCase caseObj = new JudgeCase();
            caseObj.setCaseId(caseId);
            caseObj.setIsViewed(true);
            judgeCaseMapper.updateById(caseObj);
        }
    }

    @Override
    public List<JudgeCase> getCasesByIds(List<Long> caseIds) {
        if (caseIds == null || caseIds.isEmpty()) {
            return List.of();
        }
        return judgeCaseMapper.selectList(
                Wrappers.<JudgeCase>lambdaQuery()
                        .in(JudgeCase::getCaseId, caseIds)
        );
    }
}
