package com.aijudge.judgeplaform.sse;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

@Slf4j
@Component
public class SseEmitterManager {
    private static final long SSE_TIMEOUT = TimeUnit.HOURS.toMillis(24);

    /**
     * 存储SSE连接: reviewerId -> SseEmitter
     */
    private final Map<Long, SseEmitter> emitterMap = new ConcurrentHashMap<>();

    /**
     * 创建SSE连接
     */
    public SseEmitter createConnection(Long reviewerId) {
        // 先移除旧连接
        SseEmitter oldEmitter = emitterMap.get(reviewerId);
        if (oldEmitter != null) {
            try {
                oldEmitter.complete();
            } catch (Exception e) {
                log.warn("关闭旧SSE连接失败, reviewerId={}", reviewerId);
            }
            emitterMap.remove(reviewerId);
        }

        SseEmitter emitter = new SseEmitter(SSE_TIMEOUT);
        emitterMap.put(reviewerId, emitter);

        emitter.onCompletion(() -> {
            log.info("SSE连接完成, reviewerId={}", reviewerId);
            emitterMap.remove(reviewerId);
        });

        emitter.onTimeout(() -> {
            log.info("SSE连接超时, reviewerId={}", reviewerId);
            emitterMap.remove(reviewerId);
        });

        emitter.onError(e -> {
            log.warn("SSE连接异常, reviewerId={}, error={}", reviewerId, e.getMessage());
            emitterMap.remove(reviewerId);
        });

        log.info("SSE连接已建立, reviewerId={}, 当前连接数={}", reviewerId, emitterMap.size());
        return emitter;
    }

    /**
     * 发送AI审核结果给指定审核员
     */
    public void sendAuditResult(Long reviewerId, Object data) {
        SseEmitter emitter = emitterMap.get(reviewerId);
        if (emitter == null) {
            log.warn("SSE连接不存在, reviewerId={}", reviewerId);
            return;
        }

        try {
            emitter.send(SseEmitter.event()
                    .name("audit_result")
                    .data(data));
            log.info("SSE推送成功, reviewerId={}", reviewerId);
        } catch (IOException e) {
            log.warn("SSE推送失败, reviewerId={}, error={}", reviewerId, e.getMessage());
            emitterMap.remove(reviewerId);
            try {
                emitter.complete();
            } catch (Exception ex) {
                log.warn("关闭SSE连接失败, reviewerId={}", reviewerId);
            }
        }
    }

    /**
     * 关闭指定审核员的SSE连接
     */
    public void closeConnection(Long reviewerId) {
        SseEmitter emitter = emitterMap.remove(reviewerId);
        if (emitter != null) {
            try {
                emitter.complete();
                log.info("SSE连接已关闭, reviewerId={}", reviewerId);
            } catch (Exception e) {
                log.warn("关闭SSE连接失败, reviewerId={}", reviewerId);
            }
        }
    }

    /**
     * 获取当前连接数
     */
    public int getConnectionCount() {
        return emitterMap.size();
    }
}
