-- Raw tables (Part 1). Works on MySQL 8 and MariaDB. Re-running wipes and recreates them.
DROP TABLE IF EXISTS engagement;
DROP TABLE IF EXISTS daily_metrics;
DROP TABLE IF EXISTS members;

CREATE TABLE members (
  member_id    VARCHAR(8)   NOT NULL,
  company_id   VARCHAR(4)   NOT NULL,
  age_band     VARCHAR(8)   NOT NULL,
  gender       VARCHAR(16)  NOT NULL,
  job_type     ENUM('desk','field','shift') NOT NULL,
  join_date    DATE         NOT NULL,
  PRIMARY KEY (member_id),
  KEY idx_company (company_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE daily_metrics (
  member_id        VARCHAR(8) NOT NULL,
  `date`           DATE       NOT NULL,
  steps            INT        NULL,      -- NULL when the wearable did not sync
  active_minutes   INT        NULL,
  sleep_hours      DECIMAL(4,1) NULL,
  resting_hr       INT        NULL,
  wearable_synced  ENUM('yes','no') NOT NULL,
  PRIMARY KEY (member_id, `date`),
  CONSTRAINT fk_dm_member FOREIGN KEY (member_id) REFERENCES members(member_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE engagement (
  member_id         VARCHAR(8) NOT NULL,
  `date`            DATE       NOT NULL,
  app_opens         INT        NOT NULL,
  content_views     INT        NOT NULL,
  challenge_joined  ENUM('yes','no') NOT NULL,
  mood_checkin      TINYINT    NULL,     -- 1 (very low) .. 5 (very good); NULL = no check-in that day
  PRIMARY KEY (member_id, `date`),
  CONSTRAINT fk_en_member FOREIGN KEY (member_id) REFERENCES members(member_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
