package com.aijudge.judgeplaform.controller;

import com.aijudge.judgeplaform.context.ReviewerContext;
import com.aijudge.judgeplaform.pojo.domain.R;
import com.aijudge.judgeplaform.sse.SseEmitterManager;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@Slf4j
@RestController
@RequestMapping("/sse")
@RequiredArgsConstructor
public class SseController {
    private final SseEmitterManager sseEmitterManager;

    /**
     * 建立SSE连接，用于接收AI审核结果实时推送
     * 前端通过这个SSE连接接收实时推送
     */
    @GetMapping(value = "/connect", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter connect() {
        Long reviewerId = ReviewerContext.getReviewerId();
        if (reviewerId == null) {
            log.warn("SSE连接失败：reviewerId不存在");
            return null;
        }

        SseEmitter emitter = sseEmitterManager.createConnection(reviewerId);

        // 发送初始连接成功事件
        try {
            emitter.send(SseEmitter.event()
                    .name("connected")
                    .data("SSE连接建立成功"));
        } catch (Exception e) {
            log.warn("发送SSE初始事件失败, reviewerId={}", reviewerId);
        }

        log.info("SSE连接已建立, reviewerId={}", reviewerId);
        return emitter;
    }

    /**
     * 断开SSE连接
     */
    @GetMapping("/disconnect")
    public R<Void> disconnect() {
        Long reviewerId = ReviewerContext.getReviewerId();
        if (reviewerId == null) {
            return R.fail(401, "未登录或审核员ID不存在");
        }

        sseEmitterManager.closeConnection(reviewerId);
        return R.ok("SSE连接已断开");
    }

    /**
     * 获取当前SSE连接数（用于调试）
     */
    @GetMapping("/status")
    public R<Integer> getStatus() {
        return R.ok(sseEmitterManager.getConnectionCount());
    }
}
