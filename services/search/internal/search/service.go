package search

import (
	"context"
	"database/sql"
	"errors"
	"log/slog"
	"os"
	"runtime"
	"strings"
	"sync/atomic"
	"time"
	"unicode"
	"unicode/utf8"
)

type Service struct {
	db       *sql.DB
	path     string
	file     os.FileInfo
	manifest Manifest
	config   Config
	versions Versions
	pending  chan struct{}
	slots    chan struct{}
	closed   atomic.Bool
	logger   *slog.Logger
}

func Open(ctx context.Context, path string, config Config, logger *slog.Logger) (*Service, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	file, err := os.Stat(path)
	if err != nil || !file.Mode().IsRegular() {
		return nil, failure("storage_error", "Не удалось открыть снимок.")
	}
	db, err := openDB(path, true)
	if err != nil {
		return nil, failure("storage_error", "Не удалось открыть снимок.")
	}
	db.SetMaxOpenConns(config.Workers)
	db.SetMaxIdleConns(config.Workers)
	manifest, err := readManifest(ctx, db)
	if err != nil {
		db.Close()
		return nil, failure("storage_error", "Несовместимый или повреждённый снимок; пересоберите Go-командой build.")
	}
	versions := Versions{manifest.CorpusID, manifest.CorpusRevision, manifest.IndexRevision,
		fingerprint([]any{"go-literal-unicode-or-bm25-v1", config, runtime.Version()})}
	return &Service{db: db, path: path, file: file, manifest: manifest, config: config, versions: versions,
		pending: make(chan struct{}, config.MaxPending), slots: make(chan struct{}, config.Workers), logger: logger}, nil
}
func (s *Service) Close() error { s.closed.Store(true); return s.db.Close() }
func (s *Service) Info() Info   { return Info{s.manifest, "1", s.versions.SearchRevision, s.config} }
func (s *Service) Healthy() bool {
	file, err := os.Stat(s.path)
	return !s.closed.Load() && err == nil && os.SameFile(s.file, file) && file.Size() == s.file.Size() && file.ModTime().Equal(s.file.ModTime())
}
func (s *Service) checkScope(scope Scope) error {
	if scope.CorpusID != s.manifest.CorpusID || (scope.CorpusRevision != nil && *scope.CorpusRevision != s.manifest.CorpusRevision) {
		return failure("unknown_id", "Неизвестный корпус или ревизия.")
	}
	return nil
}
func limitValue(value *int, defaultValue, maximum int) (int, error) {
	if value == nil {
		return min(defaultValue, maximum), nil
	}
	if *value < 1 || *value > maximum {
		return 0, failure("invalid_request", "Недопустимый limit.")
	}
	return *value, nil
}
func validID(value string) bool {
	return strings.TrimSpace(value) != "" && utf8.RuneCountInString(value) <= 4096 && !strings.ContainsRune(value, 0)
}
func queryExpression(query string, config Config) (string, error) {
	if !utf8.ValidString(query) || utf8.RuneCountInString(query) > config.MaxQueryChars || strings.ContainsRune(query, 0) {
		return "", failure("invalid_request", "Недопустимый запрос.")
	}
	terms := strings.FieldsFunc(query, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) })
	unique := make(map[string]bool)
	quoted := make([]string, 0, len(terms))
	for _, term := range terms {
		key := strings.ToLower(term)
		if unique[key] {
			continue
		}
		unique[key] = true
		quoted = append(quoted, `"`+term+`"`)
	}
	if len(quoted) == 0 || len(quoted) > config.MaxQueryTerms {
		return "", failure("invalid_request", "Нет слов или превышен лимит терминов.")
	}
	return strings.Join(quoted, " OR "), nil
}
func trimRunes(text string, maximum int) string {
	count := 0
	for offset := range text {
		if count == maximum {
			return text[:offset]
		}
		count++
	}
	return text
}

func (s *Service) execute(ctx context.Context, name string, operation func(context.Context, *sql.Conn) (int, error)) (err error) {
	started := time.Now()
	var queueSeconds, executionSeconds float64
	count := -1
	defer func() {
		if s.logger != nil {
			status := "ok"
			if err != nil {
				status = SafeError(err).Code
			}
			s.logger.Info("search_operation", "operation", name, "status", status, "duration_seconds", time.Since(started).Seconds(),
				"queue_seconds", queueSeconds, "execution_seconds", executionSeconds, "result_count", count,
				"corpus_id", s.versions.CorpusID, "corpus_revision", s.versions.CorpusRevision,
				"index_revision", s.versions.IndexRevision, "search_revision", s.versions.SearchRevision)
		}
	}()
	if s.closed.Load() {
		return failure("storage_error", "Сервис закрыт.")
	}
	ctx, cancel := context.WithTimeout(ctx, s.config.Timeout)
	defer cancel()
	select {
	case s.pending <- struct{}{}:
		defer func() { <-s.pending }()
	default:
		return failure("busy", "Очередь поиска заполнена.")
	}
	select {
	case s.slots <- struct{}{}:
		defer func() { <-s.slots }()
	case <-ctx.Done():
		queueSeconds = time.Since(started).Seconds()
		return ctx.Err()
	}
	queueSeconds = time.Since(started).Seconds()
	executing := time.Now()
	defer func() { executionSeconds = time.Since(executing).Seconds() }()
	if !s.Healthy() {
		return failure("storage_error", "Снимок был изменён или сервис закрыт.")
	}
	conn, err := s.db.Conn(ctx)
	if err != nil {
		return operationError(ctx, err)
	}
	defer conn.Close()
	// Проверка на том же соединении предотвращает незаметную смену ревизии в пуле.
	var raw string
	if err = conn.QueryRowContext(ctx, "SELECT value FROM metadata WHERE key='manifest'").Scan(&raw); err != nil {
		return operationError(ctx, err)
	}
	if raw != string(canonical(s.manifest)) {
		return failure("storage_error", "Манифест снимка изменился.")
	}
	count, err = operation(ctx, conn)
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return operationError(ctx, err)
}
func operationError(ctx context.Context, err error) error {
	if err == nil {
		return nil
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	var safe *Error
	if errors.As(err, &safe) {
		return safe
	}
	return failure("storage_error", "Ошибка выполнения операции с индексом.")
}
func SafeError(err error) *Error {
	if errors.Is(err, context.DeadlineExceeded) {
		return &Error{"timeout", "Истёк срок выполнения операции."}
	}
	if errors.Is(err, context.Canceled) {
		return &Error{"cancelled", "Операция отменена."}
	}
	var safe *Error
	if errors.As(err, &safe) {
		return safe
	}
	return &Error{"storage_error", "Внутренняя ошибка поискового сервиса."}
}

func (s *Service) Search(ctx context.Context, request SearchRequest) (result SearchResult, err error) {
	result = SearchResult{s.versions, []Hit{}}
	err = s.execute(ctx, "search", func(ctx context.Context, conn *sql.Conn) (int, error) {
		if err := s.checkScope(request.Scope); err != nil {
			return 0, err
		}
		limit, err := limitValue(request.Limit, 10, s.config.MaxResults)
		if err != nil {
			return 0, err
		}
		expression, err := queryExpression(request.Query, s.config)
		if err != nil {
			return 0, err
		}
		rows, err := conn.QueryContext(ctx, `SELECT d.rowid,d.doc_id,bm25(document_fts)
FROM document_fts JOIN documents d ON d.rowid=document_fts.rowid WHERE document_fts MATCH ?
ORDER BY bm25(document_fts),d.doc_id LIMIT ?`, expression, limit)
		if err != nil {
			return 0, err
		}
		type candidate struct {
			rowid int64
			id    string
			score float64
		}
		candidates := make([]candidate, 0, limit)
		for rows.Next() {
			var c candidate
			if err = rows.Scan(&c.rowid, &c.id, &c.score); err != nil {
				break
			}
			candidates = append(candidates, c)
		}
		if err == nil {
			err = rows.Err()
		}
		rows.Close()
		if err != nil {
			return 0, err
		}
		for i, c := range candidates {
			var snippet string
			if err = conn.QueryRowContext(ctx, `SELECT snippet(document_fts,1,'','',' … ',48) FROM document_fts WHERE rowid=? AND document_fts MATCH ?`, c.rowid, expression).Scan(&snippet); err != nil {
				return 0, err
			}
			result.Hits = append(result.Hits, Hit{c.id, trimRunes(snippet, s.config.SnippetChars), i + 1, c.score})
		}
		return len(result.Hits), nil
	})
	return result, err
}

func (s *Service) Read(ctx context.Context, request ReadRequest) (result TextResult, err error) {
	result.Versions = s.versions
	err = s.execute(ctx, "read", func(ctx context.Context, conn *sql.Conn) (int, error) {
		if err := s.checkScope(request.Scope); err != nil {
			return 0, err
		}
		if (request.DocID == "") == (request.ChunkID == "") || request.Offset < 0 {
			return 0, failure("invalid_request", "Нужен один id и неотрицательный offset.")
		}
		identifier := request.DocID
		if request.ChunkID != "" {
			identifier = request.ChunkID
		}
		if !validID(identifier) {
			return 0, failure("invalid_request", "Недопустимый id.")
		}
		limit, err := limitValue(request.Limit, s.config.MaxReadChars, s.config.MaxReadChars)
		if err != nil {
			return 0, err
		}
		start, total := 0, 0
		docID := request.DocID
		if request.ChunkID != "" {
			var end int
			err = conn.QueryRowContext(ctx, "SELECT doc_id,start_char,end_char FROM chunks WHERE chunk_id=?", request.ChunkID).Scan(&docID, &start, &end)
			total = end - start
			id := request.ChunkID
			result.ChunkID = &id
		} else {
			err = conn.QueryRowContext(ctx, "SELECT length(text) FROM documents WHERE doc_id=?", docID).Scan(&total)
		}
		if errors.Is(err, sql.ErrNoRows) {
			return 0, failure("unknown_id", "Неизвестный документ или фрагмент.")
		}
		if err != nil {
			return 0, err
		}
		if request.Offset > total {
			return 0, failure("invalid_request", "Offset выходит за пределы текста.")
		}
		length := min(limit, total-request.Offset)
		err = conn.QueryRowContext(ctx, "SELECT substr(text,?,?) FROM documents WHERE doc_id=?", start+request.Offset+1, length, docID).Scan(&result.Text)
		if err != nil {
			return 0, err
		}
		result.DocID = docID
		result.Offset = request.Offset
		result.TotalChars = total
		if request.Offset+length < total {
			next := request.Offset + length
			result.NextOffset = &next
		}
		return 1, nil
	})
	return result, err
}

func (s *Service) Chunks(ctx context.Context, request ChunksRequest) (result ChunksResult, err error) {
	result = ChunksResult{s.versions, []string{}, nil}
	err = s.execute(ctx, "chunks", func(ctx context.Context, conn *sql.Conn) (int, error) {
		if err := s.checkScope(request.Scope); err != nil {
			return 0, err
		}
		if !validID(request.DocID) || request.Offset < 0 {
			return 0, failure("invalid_request", "Недопустимый id или offset.")
		}
		limit, err := limitValue(request.Limit, 10, s.config.MaxResults)
		if err != nil {
			return 0, err
		}
		var exists int
		err = conn.QueryRowContext(ctx, "SELECT 1 FROM documents WHERE doc_id=?", request.DocID).Scan(&exists)
		if errors.Is(err, sql.ErrNoRows) {
			return 0, failure("unknown_id", "Неизвестный документ.")
		}
		if err != nil {
			return 0, err
		}
		rows, err := conn.QueryContext(ctx, "SELECT chunk_id FROM chunks WHERE doc_id=? ORDER BY start_char LIMIT ? OFFSET ?", request.DocID, limit+1, request.Offset)
		if err != nil {
			return 0, err
		}
		defer rows.Close()
		for rows.Next() {
			var id string
			if err = rows.Scan(&id); err != nil {
				return 0, err
			}
			result.ChunkIDs = append(result.ChunkIDs, id)
		}
		if err = rows.Err(); err != nil {
			return 0, err
		}
		if len(result.ChunkIDs) > limit {
			result.ChunkIDs = result.ChunkIDs[:limit]
			next := request.Offset + limit
			result.NextOffset = &next
		}
		return len(result.ChunkIDs), nil
	})
	return result, err
}
