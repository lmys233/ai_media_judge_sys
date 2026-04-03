CREATE TABLE IF NOT EXISTS `failure_case` (
  `failure_id` BIGINT NOT NULL COMMENT 'failure ID (snowflake)',
  `task_id` VARCHAR(64) NOT NULL COMMENT 'original task ID from audit engine',
  `trace_id` VARCHAR(64) DEFAULT NULL COMMENT 'trace ID for request tracking',
  `biz_id` BIGINT DEFAULT NULL COMMENT 'business media ID',
  `media_type` VARCHAR(16) NOT NULL DEFAULT 'text' COMMENT 'text | image | video',
  `media_url` VARCHAR(512) DEFAULT NULL COMMENT 'MinIO file URL (when image/video)',
  `content_text` TEXT DEFAULT NULL COMMENT 'text content (when media_type=text)',
  `error_message` VARCHAR(1024) NOT NULL COMMENT 'failure reason from last attempt',
  `retry_count` INT NOT NULL DEFAULT 0 COMMENT 'number of retries attempted',
  `failed_at` DATETIME NOT NULL COMMENT 'timestamp when failure occurred',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'record creation time',
  PRIMARY KEY (`failure_id`),
  INDEX `idx_task_id` (`task_id`),
  INDEX `idx_biz_id` (`biz_id`),
  INDEX `idx_failed_at` (`failed_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='audit failure records';
