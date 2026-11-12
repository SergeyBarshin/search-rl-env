package search

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"io"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"unicode"
	"unicode/utf8"

	sqlite "modernc.org/sqlite"
	sqlitelib "modernc.org/sqlite/lib"
)

type SQLInfo struct {
	DBID       string `json:"db_id"`
	DBRevision string `json:"db_revision"`
	Schema     string `json:"schema"`
}
type SQLRequest struct {
	DBID       string `json:"db_id"`
	DBRevision string `json:"db_revision"`
	Query      string `json:"query"`
}
type SQLResult struct {
	Columns    []string `json:"columns"`
	Rows       [][]any  `json:"rows"`
	Truncated  bool     `json:"truncated"`
	DBID       string   `json:"db_id"`
	DBRevision string   `json:"db_revision"`
}
type SQLStore struct {
	db   *sql.DB
	path string
	file os.FileInfo
	info SQLInfo
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (r contextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.reader.Read(p)
}

func sqlOperationError(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	var engine *sqlite.Error
	if errors.As(err, &engine) {
		switch engine.Code() & 255 {
		case sqlitelib.SQLITE_ERROR, sqlitelib.SQLITE_TOOBIG, sqlitelib.SQLITE_AUTH, sqlitelib.SQLITE_READONLY, sqlitelib.SQLITE_MISMATCH:
			return failure("sql_query_error", "SQLite не смог выполнить запрос в установленных пределах.")
		}
	}
	return failure("storage_error", "Техническая ошибка исходной SQL-базы.")
}

func OpenSQL(ctx context.Context, path, dbID string, workers int) (*SQLStore, error) {
	if !validID(dbID) || workers < 1 {
		return nil, failure("invalid_request", "Нужен db_id и положительный размер пула SQL.")
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, failure("storage_error", "Недопустимый путь SQL-снимка.")
	}
	path = absolute
	file, err := os.Open(path)
	if err != nil {
		return nil, failure("storage_error", "Не удалось открыть SQL-снимок.")
	}
	stat, err := file.Stat()
	if err != nil || !stat.Mode().IsRegular() {
		file.Close()
		return nil, failure("storage_error", "Нужен обычный файл SQL-базы.")
	}
	hash := sha256.New()
	_, err = io.Copy(hash, contextReader{ctx, file})
	file.Close()
	if err != nil {
		return nil, failure("storage_error", "Не удалось вычислить ревизию SQL-базы.")
	}
	dsn, err := sqliteURI(path, true)
	if err != nil {
		return nil, err
	}
	u, err := url.Parse(dsn)
	if err != nil {
		return nil, err
	}
	q := u.Query()
	q.Add("_pragma", "query_only(1)")
	q.Add("_pragma", "trusted_schema(0)")
	u.RawQuery = q.Encode()
	db, err := sql.Open("sqlite", u.String())
	if err != nil {
		return nil, failure("storage_error", "Не удалось открыть SQL-снимок.")
	}
	db.SetMaxOpenConns(workers)
	// Открытые idle-соединения не растут вместе с числом зарегистрированных БД.
	db.SetMaxIdleConns(0)
	rows, err := db.QueryContext(ctx, "SELECT sql FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY name")
	if err != nil {
		db.Close()
		return nil, failure("storage_error", "Не удалось прочитать SQL-схему.")
	}
	var schema strings.Builder
	for rows.Next() {
		var statement string
		if err = rows.Scan(&statement); err != nil {
			break
		}
		schema.WriteString(statement)
		schema.WriteString(";\n")
	}
	if err == nil {
		err = rows.Err()
	}
	rows.Close()
	if err != nil || schema.Len() == 0 {
		db.Close()
		return nil, failure("storage_error", "SQL-схема пуста или недоступна.")
	}
	return &SQLStore{db: db, path: path, file: stat, info: SQLInfo{dbID, hex.EncodeToString(hash.Sum(nil)), schema.String()}}, nil
}
func (s *SQLStore) Close() error  { return s.db.Close() }
func (s *SQLStore) Info() SQLInfo { return s.info }

// Разрешаем одно SELECT/WITH. Это ограниченный SQLite-интерфейс без служебных
// операторов и загрузки расширений; дополнительно действуют mode=ro/query_only.
func validateReadSQL(query string) error {
	if !utf8.ValidString(query) || strings.ContainsRune(query, 0) || len(query) > 80_000 {
		return failure("sql_query_error", "Недопустимый SQL-запрос.")
	}
	var words []string
	terminated := false
	for i := 0; i < len(query); {
		r, size := utf8.DecodeRuneInString(query[i:])
		if unicode.IsSpace(r) {
			i += size
			continue
		}
		if strings.HasPrefix(query[i:], "--") {
			if end := strings.IndexByte(query[i:], '\n'); end >= 0 {
				i += end + 1
			} else {
				break
			}
			continue
		}
		if strings.HasPrefix(query[i:], "/*") {
			end := strings.Index(query[i+2:], "*/")
			if end < 0 {
				return failure("sql_query_error", "Незавершённый комментарий SQL.")
			}
			i += end + 4
			continue
		}
		if terminated {
			return failure("sql_query_error", "Разрешено одно SQL-выражение.")
		}
		if query[i] == ';' {
			terminated = true
			i++
			continue
		}
		if strings.ContainsRune("'\"`[", r) {
			quote := byte(r)
			closeQuote := quote
			if quote == '[' {
				closeQuote = ']'
			}
			i++
			var value strings.Builder
			closed := false
			for i < len(query) {
				if query[i] == closeQuote {
					if closeQuote != ']' && i+1 < len(query) && query[i+1] == closeQuote {
						value.WriteByte(closeQuote)
						i += 2
						continue
					}
					i++
					closed = true
					break
				}
				value.WriteByte(query[i])
				i++
			}
			if !closed {
				return failure("sql_query_error", "Незавершённая строка или имя SQL.")
			}
			if quote != '\'' {
				words = append(words, strings.ToLower(value.String()))
			}
			continue
		}
		if unicode.IsLetter(r) || r == '_' {
			start := i
			i += size
			for i < len(query) {
				r, size = utf8.DecodeRuneInString(query[i:])
				if !unicode.IsLetter(r) && !unicode.IsNumber(r) && r != '_' {
					break
				}
				i += size
			}
			words = append(words, strings.ToLower(query[start:i]))
			continue
		}
		i += size
	}
	if len(words) == 0 || (words[0] != "select" && words[0] != "with") {
		return failure("sql_query_error", "Разрешены только SELECT и WITH на чтение.")
	}
	for _, word := range words {
		switch word {
		case "attach", "detach", "pragma", "vacuum", "insert", "update", "delete", "drop", "alter", "create", "replace", "reindex", "analyze", "load_extension", "readfile", "writefile", "eval", "zipfile", "fts3_tokenizer":
			return failure("sql_query_error", "Служебные операции и функции SQL запрещены.")
		}
		if strings.HasPrefix(word, "pragma_") {
			return failure("sql_query_error", "PRAGMA-функции запрещены.")
		}
	}
	return nil
}

func (s *SQLStore) Query(ctx context.Context, request SQLRequest) (result SQLResult, err error) {
	result = SQLResult{Columns: []string{}, Rows: [][]any{}, DBID: s.info.DBID, DBRevision: s.info.DBRevision}
	if request.DBID != s.info.DBID || request.DBRevision != s.info.DBRevision {
		return result, failure("unknown_id", "Неизвестная SQL-база или ревизия.")
	}
	if err = validateReadSQL(request.Query); err != nil {
		return result, err
	}
	stat, err := os.Stat(s.path)
	if err != nil || !os.SameFile(stat, s.file) || stat.Size() != s.file.Size() || !stat.ModTime().Equal(s.file.ModTime()) {
		return result, failure("storage_error", "SQL-снимок был изменён.")
	}
	conn, err := s.db.Conn(ctx)
	if err != nil {
		return result, operationError(ctx, err)
	}
	defer conn.Close()
	if _, err = sqlite.Limit(conn, sqlitelib.SQLITE_LIMIT_LENGTH, 1_000_000); err != nil {
		return result, operationError(ctx, err)
	}
	if _, err = sqlite.Limit(conn, sqlitelib.SQLITE_LIMIT_COLUMN, 100); err != nil {
		return result, operationError(ctx, err)
	}
	rows, err := conn.QueryContext(ctx, request.Query)
	if err != nil {
		return result, sqlOperationError(ctx, err)
	}
	defer rows.Close()
	result.Columns, err = rows.Columns()
	if err != nil {
		return result, operationError(ctx, err)
	}
	budget := 256 * 1024
	for rows.Next() {
		if len(result.Rows) == 100 {
			result.Truncated = true
			break
		}
		row := make([]any, len(result.Columns))
		pointers := make([]any, len(row))
		for i := range row {
			pointers[i] = &row[i]
		}
		if err = rows.Scan(pointers...); err != nil {
			return result, operationError(ctx, err)
		}
		rowBytes := 0
		for _, value := range row {
			switch value := value.(type) {
			case string:
				rowBytes += len(value)
			case []byte:
				return result, failure("sql_query_error", "Бинарные значения не поддерживаются.")
			case float64:
				if math.IsInf(value, 0) || math.IsNaN(value) {
					return result, failure("sql_query_error", "Неконечные числа не поддерживаются.")
				}
				rowBytes += 32
			default:
				rowBytes += 32
			}
		}
		if rowBytes > budget {
			result.Truncated = true
			break
		}
		budget -= rowBytes
		result.Rows = append(result.Rows, row)
	}
	if err = rows.Err(); err != nil {
		return result, sqlOperationError(ctx, err)
	}
	return result, nil
}
