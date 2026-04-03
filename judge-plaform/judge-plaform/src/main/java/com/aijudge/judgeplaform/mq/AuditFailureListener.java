package com.aijudge.judgeplaform.mq;

import com.aijudge.judgeplaform.mapper.FailureCaseMapper;
import com.aijudge.judgeplaform.pojo.domain.FailureCase;
import com.aijudge.judgeplaform.pojo.mq.FailureMessage;
import com.aijudge.judgeplaform.sse.SseEmitterManager;
import com.aijudge.judgeplaform.support.SnowflakeIdGenerator;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

@Slf4j
@Component
@RequiredArgsConstructor
public class AuditFailureListener {
    private final FailureCaseMapper failureCaseMapper;
    private final SnowflakeIdGenerator snowflakeIdGenerator;
    private final SseEmitterManager sseEmitterManager;
    private final ObjectMapper objectMapper = new ObjectMapper();

    private static final String QUEUE_NAME = "audit.failure";

    @RabbitListener(queues = QUEUE_NAME)
    public void handleAuditFailure(String messageJson) {
        log.info("Received audit failure message: {}", messageJson);
        try {
            FailureMessage failure = objectMapper.readValue(messageJson, FailureMessage.class);

            FailureCase failureCase = new FailureCase();
            failureCase.setFailureId(snowflakeIdGenerator.nextId());
            failureCase.setTaskId(failure.getTask_id());
            failureCase.setTraceId(failure.getTrace_id());
            failureCase.setBizId(failure.getBiz_id());
            failureCase.setMediaType(failure.getMedia_type());
            failureCase.setMediaUrl(failure.getMedia_url());
            failureCase.setContentText(failure.getContent_text());
            failureCase.setErrorMessage(failure.getError_message());
            failureCase.setRetryCount(3); // 最多重试3次
            failureCase.setFailedAt(parseDateTime(failure.getFailed_at()));
            failureCase.setCreatedAt(LocalDateTime.now());

            failureCaseMapper.insert(failureCase);
            log.info("Saved failure case: failureId={}, taskId={}, errorMessage={}",
                    failureCase.getFailureId(), failureCase.getTaskId(), failureCase.getErrorMessage());

            // 通知前端有审核失败
            notifyFailure(failureCase);

        } catch (Exception e) {
            log.error("Failed to process audit failure message: {}", messageJson, e);
        }
    }

    private LocalDateTime parseDateTime(String dateTimeStr) {
        if (dateTimeStr == null || dateTimeStr.isEmpty()) {
            return LocalDateTime.now();
        }
        try {
            return LocalDateTime.parse(dateTimeStr, DateTimeFormatter.ISO_DATE_TIME);
        } catch (Exception e) {
            try {
                return LocalDateTime.parse(dateTimeStr.replace(" ", "T"));
            } catch (Exception ex) {
                log.warn("Failed to parse datetime: {}", dateTimeStr);
                return LocalDateTime.now();
            }
        }
    }

    private void notifyFailure(FailureCase failureCase) {
        try {
            // 广播失败通知给所有在线审核员
            FailureNotification notification = new FailureNotification(
                    failureCase.getFailureId(),
                    failureCase.getTaskId(),
                    failureCase.getBizId(),
                    failureCase.getMediaType(),
                    failureCase.getErrorMessage()
            );
            sseEmitterManager.broadcastFailureNotification(notification);
            log.info("Failure notification broadcast: taskId={}", failureCase.getTaskId());
        } catch (Exception e) {
            log.warn("Failed to send failure notification: {}", e.getMessage());
        }
    }

    // 失败通知record
    private record FailureNotification(
            Long failureId,
            String taskId,
            Long bizId,
            String mediaType,
            String errorMessage
    ) {}
}
