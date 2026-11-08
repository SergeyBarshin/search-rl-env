// Package search реализует хранение снимков и поиск независимо от HTTP и агента.
package search

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strconv"
	"time"
	"unicode/utf8"
)

const SchemaVersion = 2
const DriverVersion = "modernc.org/sqlite@v1.44.3"
const IndexAlgorithm = "go-fts5-unicode61-document-bm25-v1"

type Error struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *Error) Error() string           { return e.Code + ": " + e.Message }
func failure(code, message string) error { return &Error{code, message} }

type Document struct {
	DocID    string `json:"doc_id"`
	Text     string `json:"text"`
	Title    string `json:"title"`
	Language string `json:"language"`
}

type Manifest struct {
	SchemaVersion  int    `json:"schema_version"`
	CorpusID       string `json:"corpus_id"`
	CorpusRevision string `json:"corpus_revision"`
	ContentHash    string `json:"content_hash"`
	ChunkChars     int    `json:"chunk_chars"`
	ChunkAlgorithm string `json:"chunk_algorithm"`
	Documents      int    `json:"documents"`
	IndexAlgorithm string `json:"index_algorithm"`
	SQLiteVersion  string `json:"sqlite_version"`
	DriverVersion  string `json:"driver_version"`
	IndexRevision  string `json:"index_revision"`
}

type Config struct {
	MaxResults    int           `json:"max_results"`
	SnippetChars  int           `json:"snippet_chars"`
	MaxReadChars  int           `json:"max_read_chars"`
	MaxQueryChars int           `json:"max_query_chars"`
	MaxQueryTerms int           `json:"max_query_terms"`
	Workers       int           `json:"workers"`
	MaxPending    int           `json:"max_pending"`
	Timeout       time.Duration `json:"timeout_nanoseconds"`
}

func DefaultConfig() Config { return Config{100, 500, 8000, 2000, 32, 4, 32, 5 * time.Second} }
func (c Config) Validate() error {
	if c.MaxResults < 1 || c.SnippetChars < 1 || c.MaxReadChars < 1 || c.MaxQueryChars < 1 || c.MaxQueryTerms < 1 || c.Workers < 1 || c.MaxPending < 1 || c.Timeout <= 0 {
		return failure("invalid_request", "Все ограничения должны быть положительными.")
	}
	return nil
}

type Versions struct {
	CorpusID       string `json:"corpus_id"`
	CorpusRevision string `json:"corpus_revision"`
	IndexRevision  string `json:"index_revision"`
	SearchRevision string `json:"search_revision"`
}
type Info struct {
	Manifest
	APIVersion     string `json:"api_version"`
	SearchRevision string `json:"search_revision"`
	Config         Config `json:"config"`
}
type Scope struct {
	CorpusID       string  `json:"corpus_id"`
	CorpusRevision *string `json:"corpus_revision"`
}
type SearchRequest struct {
	Scope
	Query string `json:"query"`
	Limit *int   `json:"limit"`
}
type Hit struct {
	DocID   string  `json:"doc_id"`
	Snippet string  `json:"snippet"`
	Rank    int     `json:"rank"`
	Score   float64 `json:"score"`
}
type SearchResult struct {
	Versions
	Hits []Hit `json:"hits"`
}
type ReadRequest struct {
	Scope
	DocID   string `json:"doc_id"`
	ChunkID string `json:"chunk_id"`
	Offset  int    `json:"offset"`
	Limit   *int   `json:"limit"`
}
type TextResult struct {
	Versions
	DocID      string  `json:"doc_id"`
	ChunkID    *string `json:"chunk_id"`
	Text       string  `json:"text"`
	Offset     int     `json:"offset"`
	NextOffset *int    `json:"next_offset"`
	TotalChars int     `json:"total_chars"`
}
type ChunksRequest struct {
	Scope
	DocID  string `json:"doc_id"`
	Offset int    `json:"offset"`
	Limit  *int   `json:"limit"`
}
type ChunksResult struct {
	Versions
	ChunkIDs   []string `json:"chunk_ids"`
	NextOffset *int     `json:"next_offset"`
}

func canonical(value any) []byte {
	var b bytes.Buffer
	e := json.NewEncoder(&b)
	e.SetEscapeHTML(false)
	if err := e.Encode(value); err != nil {
		panic(err)
	}
	return bytes.TrimSuffix(b.Bytes(), []byte("\n"))
}
func fingerprint(value any) string {
	sum := sha256.Sum256(canonical(value))
	return hex.EncodeToString(sum[:])
}

// encoding/json заменяет непарные surrogate на U+FFFD; отклоняем их до декодирования.
func validJSONUnicode(data []byte) bool {
	if !utf8.Valid(data) {
		return false
	}
	for i := 0; i < len(data); i++ {
		if data[i] != '\\' {
			continue
		}
		i++
		if i >= len(data) {
			return false
		}
		if data[i] != 'u' {
			continue
		}
		if i+4 >= len(data) {
			return false
		}
		value, err := strconv.ParseUint(string(data[i+1:i+5]), 16, 16)
		if err != nil {
			return false
		}
		switch {
		case value >= 0xd800 && value <= 0xdbff:
			if i+10 >= len(data) || data[i+5] != '\\' || data[i+6] != 'u' {
				return false
			}
			low, err := strconv.ParseUint(string(data[i+7:i+11]), 16, 16)
			if err != nil || low < 0xdc00 || low > 0xdfff {
				return false
			}
			i += 10
		case value >= 0xdc00 && value <= 0xdfff:
			return false
		default:
			i += 4
		}
	}
	return true
}
