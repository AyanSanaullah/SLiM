-- Initialize database for AI routing service
-- This script runs once when the PostgreSQL container starts

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create additional databases for testing
CREATE DATABASE ai_routing_test;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE ai_routing TO ai_user;
GRANT ALL PRIVILEGES ON DATABASE ai_routing_test TO ai_user;