-- 添加 is_viewed 字段（用于标记审核通知是否已读）
ALTER TABLE `judge_case` ADD COLUMN `is_viewed` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否已读（0=未读，1=已读）' AFTER `retry_count`;
