-- Run ONCE as a MySQL admin:   mysql -u root -p < sql/00_create_database.sql
CREATE DATABASE IF NOT EXISTS viwell CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'viwell'@'localhost'  IDENTIFIED BY 'viwell';
CREATE USER IF NOT EXISTS 'viwell'@'127.0.0.1'  IDENTIFIED BY 'viwell';
GRANT ALL PRIVILEGES ON viwell.* TO 'viwell'@'localhost';
GRANT ALL PRIVILEGES ON viwell.* TO 'viwell'@'127.0.0.1';
FLUSH PRIVILEGES;