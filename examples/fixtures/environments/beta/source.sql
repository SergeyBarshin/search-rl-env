CREATE TABLE facts (fact_id TEXT PRIMARY KEY, value TEXT NOT NULL, keyword TEXT NOT NULL);
INSERT INTO facts VALUES ('target', 'BETA-29', 'betakey');
INSERT INTO facts VALUES ('decoy', 'IGNORE', 'unrelated');
