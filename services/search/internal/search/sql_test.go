package search

import (
	"context"
	"crypto/sha256"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func sqlFixture(t *testing.T) *SQLStore {
	t.Helper()
	path := filepath.Join(t.TempDir(), "source.sqlite")
	db, err := openDB(path, false)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = db.Exec("CREATE TABLE facts(name TEXT); INSERT INTO facts VALUES ('Toronto Arenas')"); err != nil {
		t.Fatal(err)
	}
	db.Close()
	store, err := OpenSQL(context.Background(), path, "hockey", 2)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	return store
}
func sqlRequest(store *SQLStore, query string) SQLRequest {
	return SQLRequest{store.info.DBID, store.info.DBRevision, query}
}
func TestSQLReadsRealFileAndPreservesDuplicateColumns(t *testing.T) {
	store := sqlFixture(t)
	before, _ := os.ReadFile(store.path)
	result, err := store.Query(context.Background(), sqlRequest(store, "SELECT name,name FROM facts; -- valid trailing comment"))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Columns) != 2 || result.Columns[0] != "name" || result.Columns[1] != "name" || len(result.Rows) != 1 || result.Rows[0][0] != "Toronto Arenas" || result.Truncated {
		t.Fatal(result)
	}
	after, _ := os.ReadFile(store.path)
	if sha256.Sum256(before) != sha256.Sum256(after) {
		t.Fatal("source database changed")
	}
	_, err = store.Query(context.Background(), SQLRequest{DBID: "wrong", Query: "SELECT 1"})
	expectCode(t, err, "unknown_id")
}
func TestSQLRejectsWritesAttachmentsAndMultipleStatements(t *testing.T) {
	store := sqlFixture(t)
	for _, query := range []string{"DELETE FROM facts", "SELECT 1; DELETE FROM facts", "ATTACH DATABASE '/tmp/other.sqlite' AS other", "PRAGMA database_list", "SELECT * FROM pragma_database_list", `SELECT "load_extension"('file')`, `SELECT readfile('/etc/passwd')`, "WITH x AS (SELECT 1) DELETE FROM facts RETURNING name", "SELECT 1; /* ignored */ SELECT 2", "SELECT 'unfinished"} {
		_, err := store.Query(context.Background(), sqlRequest(store, query))
		expectCode(t, err, "sql_query_error")
	}
	for _, query := range []string{"SELECT 'a;b' AS text", "SELECT 1 /* ; DROP TABLE facts */; -- DELETE", "WITH x AS (SELECT 1 AS n) SELECT n FROM x"} {
		if _, err := store.Query(context.Background(), sqlRequest(store, query)); err != nil {
			t.Fatal(query, err)
		}
	}
	_, err := store.Query(context.Background(), sqlRequest(store, "SELECT no_such_column FROM facts"))
	expectCode(t, err, "sql_query_error")
}
func TestSQLBoundsAndCancellation(t *testing.T) {
	store := sqlFixture(t)
	result, err := store.Query(context.Background(), sqlRequest(store, "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<1000) SELECT x FROM n"))
	if err != nil || len(result.Rows) != 100 || !result.Truncated {
		t.Fatal(result, err)
	}
	_, err = store.Query(context.Background(), sqlRequest(store, "SELECT zeroblob(100)"))
	expectCode(t, err, "sql_query_error")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	_, err = store.Query(ctx, sqlRequest(store, "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<100000000) SELECT sum(x) FROM n"))
	if err == nil || ctx.Err() == nil {
		t.Fatal("SQL deadline not respected", err)
	}
}
