package com.aijudge.judgeplaform.config;

import com.aijudge.judgeplaform.context.ReviewerContext;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

@Component
public class ReviewerContextFilter extends OncePerRequestFilter {
    private static final Long FIXED_REVIEWER_ID = 1001L;

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
            throws ServletException, IOException {
        try {
            ReviewerContext.setReviewerId(FIXED_REVIEWER_ID);
            filterChain.doFilter(request, response);
        } finally {
            ReviewerContext.clear();
        }
    }
}
