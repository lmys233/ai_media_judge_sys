package com.aijudge.judgeplaform.context;

public final class ReviewerContext {
    private static final ThreadLocal<Long> REVIEWER_HOLDER = new ThreadLocal<>();

    private ReviewerContext() {
    }

    public static void setReviewerId(Long reviewerId) {
        REVIEWER_HOLDER.set(reviewerId);
    }

    public static Long getReviewerId() {
        return REVIEWER_HOLDER.get();
    }

    public static void clear() {
        REVIEWER_HOLDER.remove();
    }
}
