CREATE TABLE IF NOT EXISTS `judge_case` (
  `case_id` BIGINT NOT NULL COMMENT 'case ID (snowflake)',
  `media_id` BIGINT DEFAULT NULL COMMENT 'business media ID',
  `content_text` TEXT DEFAULT NULL COMMENT 'text content (when media_type=text)',
  `media_type` VARCHAR(16) NOT NULL DEFAULT 'text' COMMENT 'text | image | video',
  `media_url` VARCHAR(512) DEFAULT NULL COMMENT 'MinIO file URL (when image/video)',
  `violation_types` JSON NOT NULL COMMENT 'violation type array',
  `review_reason` VARCHAR(50) NOT NULL COMMENT 'review reason',
  `evidence` JSON NOT NULL COMMENT 'evidence array',
  `source` VARCHAR(16) NOT NULL DEFAULT 'human' COMMENT 'human | ai',
  `reviewer_id` BIGINT NOT NULL COMMENT 'reviewer ID',
  `review_time` DATETIME NOT NULL COMMENT 'review time',
  PRIMARY KEY (`case_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='manual review result';
