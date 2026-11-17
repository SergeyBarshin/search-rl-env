CREATE TABLE facts (fact_id TEXT PRIMARY KEY, value TEXT NOT NULL, keyword TEXT NOT NULL);
INSERT INTO facts VALUES ('target', 'ALPHA-17', 'alphakey');
INSERT INTO facts VALUES ('decoy', 'IGNORE', 'unrelated');
