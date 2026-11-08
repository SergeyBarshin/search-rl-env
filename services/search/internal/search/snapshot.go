package search

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	_ "modernc.org/sqlite"
)

const maxDocumentChars = 1_000_000

func sqliteURI(path string, readonly bool) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	u := url.URL{Scheme: "file", Path: filepath.ToSlash(abs)}
	q := url.Values{}
	if readonly {
		q.Set("mode", "ro")
	}
	q.Add("_pragma", "foreign_keys(1)")
	u.RawQuery = q.Encode()
	return u.String(), nil
}
func openDB(path string, readonly bool) (*sql.DB, error) {
	u, err := sqliteURI(path, readonly)
	if err != nil {
		return nil, err
	}
	return sql.Open("sqlite", u)
}
func readManifest(ctx context.Context, db *sql.DB) (Manifest, error) {
	var text string
	var m Manifest
	if err := db.QueryRowContext(ctx, "SELECT value FROM metadata WHERE key='manifest'").Scan(&text); err != nil {
		return m, err
	}
	if err := json.Unmarshal([]byte(text), &m); err != nil {
		return m, err
	}
	if m.SchemaVersion != SchemaVersion || m.IndexAlgorithm != IndexAlgorithm || m.DriverVersion != DriverVersion || m.Documents < 1 || m.CorpusID == "" || m.CorpusRevision == "" || m.IndexRevision == "" {
		return m, failure("storage_error", "Несовместимый снимок; пересоберите индекс Go-командой build.")
	}
	var version string
	if err := db.QueryRowContext(ctx, "SELECT sqlite_version()").Scan(&version); err != nil {
		return m, err
	}
	if version != m.SQLiteVersion {
		return m, failure("storage_error", "Версия SQLite отличается от версии сборки.")
	}
	return m, nil
}

// Build читает JSONL потоком и публикует готовый файл без перезаписи.
func Build(ctx context.Context, input io.Reader, path, corpusID, revision string, chunkChars int) (manifest Manifest, err error) {
	if strings.TrimSpace(corpusID) == "" || chunkChars < 1 || chunkChars > maxDocumentChars || (revision != "" && strings.TrimSpace(revision) == "") {
		return manifest, failure("invalid_request", "Недопустимые параметры корпуса.")
	}
	if err = os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return manifest, failure("storage_error", "Не удалось создать каталог снимка.")
	}
	temp, err := os.CreateTemp(filepath.Dir(path), ".search-build-*.sqlite")
	if err != nil {
		return manifest, failure("storage_error", "Не удалось создать временный снимок.")
	}
	temporary := temp.Name()
	temp.Close()
	defer os.Remove(temporary)
	db, err := openDB(temporary, false)
	if err != nil {
		return manifest, failure("storage_error", "Не удалось открыть временный снимок.")
	}
	defer db.Close()
	db.SetMaxOpenConns(1)
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return manifest, failure("storage_error", "Не удалось начать сборку.")
	}
	defer tx.Rollback()
	_, err = tx.ExecContext(ctx, `
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE documents(doc_id TEXT NOT NULL UNIQUE, text TEXT NOT NULL, title TEXT NOT NULL, language TEXT NOT NULL);
CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id), start_char INTEGER NOT NULL, end_char INTEGER NOT NULL);
CREATE INDEX chunks_document ON chunks(doc_id, start_char);
CREATE VIRTUAL TABLE document_fts USING fts5(title, text, content='documents', content_rowid='rowid', tokenize='unicode61');`)
	if err != nil {
		return manifest, failure("storage_error", "Не удалось создать FTS5-индекс.")
	}
	docStmt, err := tx.PrepareContext(ctx, "INSERT INTO documents VALUES (?, ?, ?, ?)")
	if err != nil {
		return manifest, failure("storage_error", "Не удалось подготовить загрузку.")
	}
	defer docStmt.Close()
	chunkStmt, err := tx.PrepareContext(ctx, "INSERT INTO chunks VALUES (?, ?, ?, ?)")
	if err != nil {
		return manifest, failure("storage_error", "Не удалось подготовить фрагменты.")
	}
	defer chunkStmt.Close()
	scanner := bufio.NewScanner(input)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	for scanner.Scan() {
		if err := ctx.Err(); err != nil {
			return manifest, err
		}
		line := scanner.Bytes()
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		if !validJSONUnicode(line) {
			return manifest, failure("invalid_request", "Некорректный UTF-8 во входном JSONL.")
		}
		var d Document
		decoder := json.NewDecoder(bytes.NewReader(line))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&d); err != nil {
			return manifest, failure("invalid_request", "Неверная запись корпуса.")
		}
		var extra any
		if decoder.Decode(&extra) != io.EOF {
			return manifest, failure("invalid_request", "Ожидалась одна JSON-запись в строке.")
		}
		if d.Language == "" {
			d.Language = "und"
		}
		if strings.TrimSpace(d.DocID) == "" || strings.TrimSpace(d.Text) == "" || strings.TrimSpace(d.Language) == "" || utf8.RuneCountInString(d.DocID) > 4096 || utf8.RuneCountInString(d.Language) > 64 || utf8.RuneCountInString(d.Text) > maxDocumentChars || utf8.RuneCountInString(d.Title) > maxDocumentChars || strings.ContainsRune(d.DocID+d.Text+d.Title+d.Language, 0) {
			return manifest, failure("invalid_request", "Недопустимые поля или размер документа.")
		}
		if _, err := docStmt.ExecContext(ctx, d.DocID, d.Text, d.Title, d.Language); err != nil {
			var exists int
			if tx.QueryRowContext(ctx, "SELECT 1 FROM documents WHERE doc_id=?", d.DocID).Scan(&exists) == nil {
				return manifest, failure("invalid_request", "Повторный doc_id во входном корпусе.")
			}
			return manifest, failure("storage_error", "Не удалось сохранить документ.")
		}
		textHash := fingerprint(d.Text)
		runes := utf8.RuneCountInString(d.Text)
		for start := 0; start < runes; start += chunkChars {
			end := min(start+chunkChars, runes)
			chunkID := fingerprint([]any{d.DocID, textHash, "chars-v1", chunkChars, start, end})
			if _, err := chunkStmt.ExecContext(ctx, chunkID, d.DocID, start, end); err != nil {
				return manifest, failure("storage_error", "Не удалось сохранить фрагмент.")
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return manifest, failure("invalid_request", "Ошибка чтения JSONL или превышен размер строки.")
	}
	rows, err := tx.QueryContext(ctx, "SELECT doc_id,text,title,language FROM documents ORDER BY doc_id")
	if err != nil {
		return manifest, failure("storage_error", "Не удалось вычислить версию корпуса.")
	}
	digest := sha256.New()
	count := 0
	for rows.Next() {
		var id, text, title, language string
		if err = rows.Scan(&id, &text, &title, &language); err != nil {
			break
		}
		digest.Write(canonical([]string{id, text, title, language}))
		digest.Write([]byte("\n"))
		count++
	}
	if err == nil {
		err = rows.Err()
	}
	rows.Close()
	if err != nil {
		return manifest, failure("storage_error", "Не удалось вычислить версию корпуса.")
	}
	if count == 0 {
		return manifest, failure("invalid_request", "Пустой корпус не поддерживается.")
	}
	contentHash := hex.EncodeToString(digest.Sum(nil))
	if revision == "" {
		revision = contentHash
	}
	var sqliteVersion string
	if err = tx.QueryRowContext(ctx, "SELECT sqlite_version()").Scan(&sqliteVersion); err != nil {
		return manifest, err
	}
	manifest = Manifest{SchemaVersion, corpusID, revision, contentHash, chunkChars, "chars-v1", count, IndexAlgorithm, sqliteVersion, DriverVersion, ""}
	manifest.IndexRevision = fingerprint(manifest)
	for _, statement := range []string{"INSERT INTO document_fts(document_fts) VALUES ('rebuild')", "INSERT INTO document_fts(document_fts,rank) VALUES ('integrity-check',1)"} {
		if _, err = tx.ExecContext(ctx, statement); err != nil {
			return manifest, failure("storage_error", "Индекс не прошёл проверку FTS5.")
		}
	}
	if _, err = tx.ExecContext(ctx, "INSERT INTO metadata VALUES ('manifest',?)", string(canonical(manifest))); err != nil {
		return manifest, err
	}
	if err = tx.Commit(); err != nil {
		return manifest, failure("storage_error", "Не удалось завершить сборку.")
	}
	var integrity string
	if err = db.QueryRowContext(ctx, "PRAGMA integrity_check").Scan(&integrity); err != nil || integrity != "ok" {
		return manifest, failure("storage_error", "Снимок не прошёл проверку целостности.")
	}
	if err = db.Close(); err != nil {
		return manifest, err
	}
	file, err := os.Open(temporary)
	if err != nil {
		return manifest, err
	}
	err = file.Sync()
	file.Close()
	if err != nil {
		return manifest, err
	}
	if err = os.Link(temporary, path); err != nil {
		if !errors.Is(err, os.ErrExist) {
			return manifest, failure("storage_error", "Не удалось опубликовать снимок.")
		}
		existing, openErr := openDB(path, true)
		if openErr != nil {
			return manifest, failure("storage_error", "Не удалось проверить существующий снимок.")
		}
		defer existing.Close()
		previous, readErr := readManifest(ctx, existing)
		if readErr != nil {
			return manifest, failure("snapshot_conflict", "Путь занят несовместимым снимком.")
		}
		if previous != manifest {
			return manifest, failure("snapshot_conflict", "По этому пути опубликован другой снимок.")
		}
	}
	return manifest, nil
}
