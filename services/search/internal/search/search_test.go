package search

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

var testDocuments = []Document{
	{DocID: "ru", Text: "Поиск документов. Срок перенесён на ноябрь. Русский текст 😀.", Language: "ru"},
	{DocID: "en", Text: "Equipment delivery postponed until November.", Language: "en"},
	{DocID: "other", Text: "The library opens on Monday."},
}

func input(docs []Document) io.Reader {
	var b bytes.Buffer
	for _, d := range docs {
		json.NewEncoder(&b).Encode(d)
	}
	return &b
}
func fixture(t *testing.T, config Config) (*Service, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "corpus.sqlite")
	if _, err := Build(context.Background(), input(testDocuments), path, "demo", "", 17); err != nil {
		t.Fatal(err)
	}
	s, err := Open(context.Background(), path, config, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s, path
}
func expectCode(t *testing.T, err error, code string) {
	t.Helper()
	if err == nil || SafeError(err).Code != code {
		t.Fatalf("expected %s, got %v", code, err)
	}
}
func number(n int) *int { return &n }
func scope() Scope      { return Scope{CorpusID: "demo"} }

func TestSnapshotReproducibilityAndConflicts(t *testing.T) {
	s, path := fixture(t, DefaultConfig())
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	reversed := []Document{testDocuments[2], testDocuments[1], testDocuments[0]}
	m, err := Build(context.Background(), input(reversed), path, "demo", "", 17)
	if err != nil {
		t.Fatal(err)
	}
	if m != s.manifest {
		t.Fatal("input order changed manifest")
	}
	after, _ := os.ReadFile(path)
	if !bytes.Equal(before, after) {
		t.Fatal("published snapshot was rewritten")
	}
	_, err = Build(context.Background(), input(testDocuments[:1]), path, "demo", "", 17)
	expectCode(t, err, "snapshot_conflict")
	other, err := Build(context.Background(), input(reversed), filepath.Join(t.TempDir(), "other.sqlite"), "demo", "", 9)
	if err != nil {
		t.Fatal(err)
	}
	if other.CorpusRevision != m.CorpusRevision || other.IndexRevision == m.IndexRevision {
		t.Fatal("content and index revisions are not independent")
	}
}

func TestInvalidInputDoesNotPublish(t *testing.T) {
	cases := []string{"", `null`, `{"doc_id":"x","text":""}`, `{"doc_id":"x","text":"a","unknown":1}`,
		`{"doc_id":"x","text":"a"} {}`, `{"doc_id":"x","text":"a\u0000b"}`, "{broken}",
		`{"doc_id":"x","text":"\ud800"}`, `{"doc_id":"x","text":"\udc00"}`,
		"{\"doc_id\":\"x\",\"text\":\"\xff\"}", "{\"doc_id\":\"x\",\"text\":\"a\"}\n{\"doc_id\":\"x\",\"text\":\"a\"}"}
	for i, data := range cases {
		t.Run(string(rune('a'+i)), func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "bad.sqlite")
			_, err := Build(context.Background(), strings.NewReader(data), path, "demo", "", 17)
			expectCode(t, err, "invalid_request")
			entries, _ := os.ReadDir(dir)
			if len(entries) != 0 {
				t.Fatal("partial snapshot left after failed build")
			}
		})
	}
	data := input([]Document{{DocID: "x", Text: strings.Repeat("a", maxDocumentChars+1)}})
	_, err := Build(context.Background(), data, filepath.Join(t.TempDir(), "large.sqlite"), "demo", "", 17)
	expectCode(t, err, "invalid_request")
}

func TestConcurrentBuilders(t *testing.T) {
	path := filepath.Join(t.TempDir(), "same.sqlite")
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := Build(context.Background(), input(testDocuments), path, "demo", "", 17)
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestEscapedUnicodePreservesOriginalText(t *testing.T) {
	path := filepath.Join(t.TempDir(), "unicode.sqlite")
	_, err := Build(context.Background(), strings.NewReader(`{"doc_id":"x","text":"\ud83d\ude00\\u literal"}`), path, "demo", "", 2)
	if err != nil {
		t.Fatal(err)
	}
	s, err := Open(context.Background(), path, DefaultConfig(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	result, err := s.Read(context.Background(), ReadRequest{Scope: scope(), DocID: "x"})
	if err != nil || result.Text != "😀\\u literal" {
		t.Fatal(result, err)
	}
}

func TestSearchReadAndUnicode(t *testing.T) {
	s, _ := fixture(t, DefaultConfig())
	ctx := context.Background()
	for query, id := range map[string]string{"перенесён": "ru", "POSTPONED": "en", `postponed'; DROP TABLE documents; --`: "en"} {
		result, err := s.Search(ctx, SearchRequest{Scope: scope(), Query: query})
		if err != nil {
			t.Fatal(err)
		}
		if len(result.Hits) != 1 || result.Hits[0].DocID != id || result.Hits[0].Rank != 1 {
			t.Fatalf("unexpected results: %+v", result)
		}
	}
	empty, err := s.Search(ctx, SearchRequest{Scope: scope(), Query: "notincorpus"})
	if err != nil || empty.Hits == nil || len(empty.Hits) != 0 {
		t.Fatal(empty, err)
	}
	chunks, err := s.Chunks(ctx, ChunksRequest{Scope: scope(), DocID: "ru", Limit: number(100)})
	if err != nil {
		t.Fatal(err)
	}
	var text strings.Builder
	for _, id := range chunks.ChunkIDs {
		result, err := s.Read(ctx, ReadRequest{Scope: scope(), ChunkID: id})
		if err != nil {
			t.Fatal(err)
		}
		text.WriteString(result.Text)
	}
	if text.String() != testDocuments[0].Text {
		t.Fatal("chunks changed original Unicode text")
	}
	first, err := s.Read(ctx, ReadRequest{Scope: scope(), DocID: "ru", Limit: number(5)})
	if err != nil {
		t.Fatal(err)
	}
	second, err := s.Read(ctx, ReadRequest{Scope: scope(), DocID: "ru", Offset: *first.NextOffset, Limit: number(5)})
	if err != nil {
		t.Fatal(err)
	}
	if first.Text+second.Text != string([]rune(testDocuments[0].Text)[:10]) {
		t.Fatal("offset is not measured in Unicode code points")
	}
	end, err := s.Read(ctx, ReadRequest{Scope: scope(), DocID: "ru", Offset: len([]rune(testDocuments[0].Text))})
	if err != nil || end.Text != "" || end.NextOffset != nil {
		t.Fatal(end, err)
	}
	page, err := s.Chunks(ctx, ChunksRequest{Scope: scope(), DocID: "ru", Limit: number(1)})
	if err != nil || page.NextOffset == nil || *page.NextOffset != 1 {
		t.Fatal(page, err)
	}
}

func TestRankingAndReadableTitleMatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rank.sqlite")
	docs := []Document{{DocID: "b", Text: "needle"}, {DocID: "a", Text: "needle"}, {DocID: "c", Text: strings.Repeat("hay ", 100) + "needle"}, {DocID: "title", Text: "Body of document.", Title: "uniquetitle"}}
	if _, err := Build(context.Background(), input(docs), path, "demo", "", 5); err != nil {
		t.Fatal(err)
	}
	cfg := DefaultConfig()
	cfg.SnippetChars = 4
	s, err := Open(context.Background(), path, cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	result, err := s.Search(context.Background(), SearchRequest{Scope: scope(), Query: "needle", Limit: number(2)})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Hits) != 2 || result.Hits[0].DocID != "a" || result.Hits[1].DocID != "b" || len([]rune(result.Hits[0].Snippet)) > 4 {
		t.Fatal(result)
	}
	title, err := s.Search(context.Background(), SearchRequest{Scope: scope(), Query: "uniquetitle"})
	if err != nil {
		t.Fatal(err)
	}
	if title.Hits[0].Snippet != "Body" {
		t.Fatal("snippet did not come from readable body")
	}
}

func TestInvalidRequests(t *testing.T) {
	s, _ := fixture(t, DefaultConfig())
	ctx := context.Background()
	for _, query := range []string{"", "   ", "()", "\x00", strings.Repeat("a", 2001)} {
		_, err := s.Search(ctx, SearchRequest{Scope: scope(), Query: query})
		expectCode(t, err, "invalid_request")
	}
	for _, limit := range []int{-1, 0, 101} {
		_, err := s.Search(ctx, SearchRequest{Scope: scope(), Query: "text", Limit: number(limit)})
		expectCode(t, err, "invalid_request")
	}
	_, err := s.Search(ctx, SearchRequest{Scope: Scope{CorpusID: "wrong"}, Query: "text"})
	expectCode(t, err, "unknown_id")
	revision := "wrong"
	_, err = s.Search(ctx, SearchRequest{Scope: Scope{CorpusID: "demo", CorpusRevision: &revision}, Query: "text"})
	expectCode(t, err, "unknown_id")
	for _, request := range []ReadRequest{{Scope: scope(), DocID: "ru", Offset: -1}, {Scope: scope(), DocID: "ru", Offset: 10000}, {Scope: scope(), DocID: "ru", ChunkID: "chunk"}, {Scope: scope()}} {
		_, err = s.Read(ctx, request)
		expectCode(t, err, "invalid_request")
	}
	_, err = s.Read(ctx, ReadRequest{Scope: scope(), DocID: "missing"})
	expectCode(t, err, "unknown_id")
	_, err = s.Read(ctx, ReadRequest{Scope: scope(), ChunkID: "missing"})
	expectCode(t, err, "unknown_id")
	_, err = s.Chunks(ctx, ChunksRequest{Scope: scope(), DocID: "missing"})
	expectCode(t, err, "unknown_id")
}

func TestSixteenParallelRequests(t *testing.T) {
	s, _ := fixture(t, DefaultConfig())
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 5 {
				result, err := s.Search(context.Background(), SearchRequest{Scope: scope(), Query: "postponed"})
				if err != nil {
					t.Error(err)
					return
				}
				if len(result.Hits) != 1 || result.Hits[0].DocID != "en" {
					t.Error(result)
				}
			}
		}()
	}
	wg.Wait()
}

func TestQueueDeadlineAndCancellation(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Workers = 1
	cfg.MaxPending = 1
	s, _ := fixture(t, cfg)
	// Занимаем реальное соединение пула: запрос ожидает его, не блокируя вызывающий поток.
	conn, err := s.db.Conn(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { _, err := s.Search(ctx, SearchRequest{Scope: scope(), Query: "text"}); done <- err }()
	deadline := time.After(2 * time.Second)
	for len(s.pending) == 0 {
		select {
		case <-deadline:
			t.Fatal("request did not enter queue")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	_, err = s.Search(context.Background(), SearchRequest{Scope: scope(), Query: "text"})
	expectCode(t, err, "busy")
	cancel()
	select {
	case err = <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancel did not release request")
	}
	ctx, stop := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer stop()
	_, err = s.Search(ctx, SearchRequest{Scope: scope(), Query: "text"})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	conn.Close()
	if _, err = s.Search(context.Background(), SearchRequest{Scope: scope(), Query: "postponed"}); err != nil {
		t.Fatal(err)
	}
}

func TestDriverInterruptsExecutingQuery(t *testing.T) {
	s, _ := fixture(t, DefaultConfig())
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	var sum int64
	err := s.db.QueryRowContext(ctx, `WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<100000000) SELECT sum(x) FROM n`).Scan(&sum)
	if err == nil || ctx.Err() == nil {
		t.Fatal("SQLite did not honour deadline")
	}
	if _, err = s.Search(context.Background(), SearchRequest{Scope: scope(), Query: "postponed"}); err != nil {
		t.Fatal(err)
	}
}

func TestMissingReplacedAndClosedSnapshot(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing.sqlite")
	_, err := Open(context.Background(), path, DefaultConfig(), nil)
	expectCode(t, err, "storage_error")
	if _, err = os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("missing file was created")
	}
	s, path := fixture(t, DefaultConfig())
	if err = os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, err = Build(context.Background(), input(testDocuments[:1]), path, "demo", "", 17); err != nil {
		t.Fatal(err)
	}
	_, err = s.Search(context.Background(), SearchRequest{Scope: scope(), Query: "text"})
	expectCode(t, err, "storage_error")
	s.Close()
	_, err = s.Search(context.Background(), SearchRequest{Scope: scope(), Query: "text"})
	expectCode(t, err, "storage_error")
}

func TestHTTPContractAndSanitizedLogs(t *testing.T) {
	s, _ := fixture(t, DefaultConfig())
	var logs bytes.Buffer
	s.logger = slog.New(slog.NewJSONHandler(&logs, nil))
	handler := s.Handler()
	cases := []struct {
		path, body string
		status     int
		code       string
	}{
		{"/v1/search", `{"corpus_id":"demo","query":"postponed"}`, 200, ""},
		{"/v1/search", `{"corpus_id":"demo","query":"private-query","limit":0}`, 400, "invalid_request"},
		{"/v1/search", `{"corpus_id":"demo","query":"text","unknown":1}`, 400, "invalid_request"},
		{"/v1/search", `{"corpus_id":"demo","query":"text"} {}`, 400, "invalid_request"},
		{"/v1/read", `{"corpus_id":"demo","doc_id":"private-id"}`, 404, "unknown_id"},
		{"/v1/search", strings.Repeat(" ", 64*1024+1), 400, "invalid_request"},
	}
	for _, c := range cases {
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, httptest.NewRequest("POST", c.path, strings.NewReader(c.body)))
		if w.Code != c.status {
			t.Fatalf("status %d: %s", w.Code, w.Body.String())
		}
		if c.code != "" {
			var body struct {
				Error Error `json:"error"`
			}
			if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil || body.Error.Code != c.code {
				t.Fatal(w.Body.String(), err)
			}
		}
	}
	if strings.Contains(logs.String(), "private-") {
		t.Fatal("logs contain query or identifier")
	}
	for _, path := range []string{"/healthz", "/v1/manifest"} {
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, httptest.NewRequest("GET", path, nil))
		if w.Code != 200 {
			t.Fatal(w.Code, w.Body.String())
		}
	}
}

func BenchmarkSearch(b *testing.B) {
	for _, count := range []int{1000, 10000} {
		b.Run(fmt.Sprintf("documents-%d", count), func(b *testing.B) {
			b.StopTimer()
			var corpus bytes.Buffer
			for i := range count {
				text := strings.Repeat("Equipment delivery postponed until November. ", 45)
				if i%2 == 1 {
					text = strings.Repeat("Поставка оборудования перенесена на ноябрь. ", 45)
				}
				json.NewEncoder(&corpus).Encode(Document{DocID: fmt.Sprintf("doc-%08d", i), Text: text})
			}
			path := filepath.Join(b.TempDir(), "bench.sqlite")
			if _, err := Build(context.Background(), &corpus, path, "demo", "", 2000); err != nil {
				b.Fatal(err)
			}
			s, err := Open(context.Background(), path, DefaultConfig(), nil)
			if err != nil {
				b.Fatal(err)
			}
			defer s.Close()
			request := SearchRequest{Scope: scope(), Query: "equipment"}
			if _, err = s.Search(context.Background(), request); err != nil {
				b.Fatal(err)
			}
			b.StartTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					if _, err := s.Search(context.Background(), request); err != nil {
						b.Error(err)
						return
					}
				}
			})
		})
	}
}
