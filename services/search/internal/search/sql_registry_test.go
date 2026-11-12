package search

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

func registryFixture(t *testing.T) (*SQLRegistry, string, map[SQLDatabase]string) {
	t.Helper()
	dir := t.TempDir()
	catalog := SQLCatalog{Version: 1}
	expected := make(map[SQLDatabase]string)
	for i := range 4 {
		path := filepath.Join(dir, fmt.Sprintf("source-%d.sqlite", i))
		db, err := openDB(path, false)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = db.Exec("CREATE TABLE routing_probe(value TEXT)"); err != nil {
			t.Fatal(err)
		}
		label := fmt.Sprintf("database-%d", i)
		if _, err = db.Exec("INSERT INTO routing_probe VALUES (?)", label); err != nil {
			t.Fatal(err)
		}
		db.Close()
		id := fmt.Sprintf("db-%d", i)
		if i < 2 {
			id = "two-revisions"
		}
		store, err := OpenSQL(context.Background(), path, id, 16)
		if err != nil {
			t.Fatal(err)
		}
		info := store.Info()
		store.Close()
		catalog.Databases = append(catalog.Databases, SQLCatalogSpec{id, filepath.Base(path), info.DBRevision})
		expected[SQLDatabase{id, info.DBRevision}] = label
	}
	path := filepath.Join(dir, "catalog.json")
	if err := os.WriteFile(path, canonical(catalog), 0600); err != nil {
		t.Fatal(err)
	}
	registry, err := LoadSQLRegistry(context.Background(), path, 16)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { registry.Close() })
	return registry, path, expected
}

func TestRegistryRoutingAndDiscovery(t *testing.T) {
	registry, _, expected := registryFixture(t)
	if len(registry.Databases()) != 4 {
		t.Fatal("not all snapshots registered")
	}
	_, err := registry.Resolve("two-revisions", "")
	expectCode(t, err, "invalid_request")
	_, err = registry.Resolve("", "")
	expectCode(t, err, "invalid_request")
	_, err = registry.Resolve("../../other", "missing")
	expectCode(t, err, "unknown_id")
	for key, value := range expected {
		store, err := registry.Resolve(key.DBID, key.DBRevision)
		if err != nil {
			t.Fatal(err)
		}
		result, err := store.Query(context.Background(), SQLRequest{key.DBID, key.DBRevision, "SELECT value FROM routing_probe"})
		if err != nil || len(result.Rows) != 1 || result.Rows[0][0] != value {
			t.Fatal(result, err)
		}
		if store.db.Stats().OpenConnections != 0 {
			t.Fatal("idle connection retained")
		}
	}
	service, err := NewSQLService(DefaultConfig(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer service.Close()
	handler := service.HandlerWithRegistry(registry)
	for _, endpoint := range []string{"/healthz", "/v1/sql/databases"} {
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, httptest.NewRequest("GET", endpoint, nil))
		if w.Code != 200 {
			t.Fatal(w.Code, w.Body.String())
		}
		if bytes.Contains(w.Body.Bytes(), []byte("source-")) {
			t.Fatal("catalog exposed filesystem paths")
		}
	}
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, httptest.NewRequest("GET", "/v1/sql/schema?db_id=two-revisions", nil))
	if w.Code != 400 {
		t.Fatal(w.Code)
	}
	w = httptest.NewRecorder()
	handler.ServeHTTP(w, httptest.NewRequest("POST", "/v1/search", bytes.NewBufferString(`{}`)))
	if w.Code != 404 {
		t.Fatal("SQL-only unexpectedly exposes search")
	}
}

func TestRegistryRejectsWrongHashDuplicateAndInvalidCatalog(t *testing.T) {
	_, path, _ := registryFixture(t)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var original SQLCatalog
	if err = json.Unmarshal(data, &original); err != nil {
		t.Fatal(err)
	}
	wrong := original
	wrong.Databases = append([]SQLCatalogSpec(nil), original.Databases...)
	wrong.Databases[0].Revision = fmt.Sprintf("%064d", 0)
	duplicate := original
	duplicate.Databases = append(append([]SQLCatalogSpec(nil), original.Databases...), original.Databases[0])
	for _, catalog := range []SQLCatalog{wrong, duplicate, {Version: 2, Databases: original.Databases}, {Version: 1}} {
		if err = os.WriteFile(path, canonical(catalog), 0600); err != nil {
			t.Fatal(err)
		}
		registry, err := LoadSQLRegistry(context.Background(), path, 16)
		if err == nil {
			registry.Close()
			t.Fatal("invalid catalog accepted")
		}
	}
}

func TestRegistry1000OperationsAt16And32And64(t *testing.T) {
	registry, _, expected := registryFixture(t)
	config := DefaultConfig()
	service, err := NewSQLService(config, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer service.Close()
	handler := service.HandlerWithRegistry(registry)
	databases := registry.Databases()
	for _, concurrency := range []int{16, 32, 64} {
		t.Run(fmt.Sprintf("concurrency-%d", concurrency), func(t *testing.T) {
			var next, successful atomic.Int64
			var wg sync.WaitGroup
			for range concurrency {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for {
						index := int(next.Add(1) - 1)
						if index >= 1000 {
							return
						}
						key := databases[index%len(databases)]
						request := SQLRequest{key.DBID, key.DBRevision, "SELECT value FROM routing_probe"}
						w := httptest.NewRecorder()
						handler.ServeHTTP(w, httptest.NewRequest("POST", "/v1/sql", bytes.NewReader(canonical(request))))
						var result SQLResult
						if w.Code != 200 || json.Unmarshal(w.Body.Bytes(), &result) != nil || len(result.Rows) != 1 || result.Rows[0][0] != expected[key] || result.DBID != key.DBID || result.DBRevision != key.DBRevision {
							t.Errorf("wrong database/result: status=%d, body=%s", w.Code, w.Body.String())
							return
						}
						successful.Add(1)
					}
				}()
			}
			wg.Wait()
			if successful.Load() != 1000 {
				t.Fatalf("only %d operations passed", successful.Load())
			}
			for _, store := range registry.stores {
				if store.db.Stats().OpenConnections != 0 {
					t.Fatal("idle SQL connection retained")
				}
			}
		})
	}
}
