package search

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"sort"
)

type SQLDatabase struct {
	DBID       string `json:"db_id"`
	DBRevision string `json:"db_revision"`
}
type SQLCatalogSpec struct {
	DBID     string `json:"db_id"`
	Path     string `json:"path"`
	Revision string `json:"revision"`
}
type SQLCatalog struct {
	Version   int              `json:"version"`
	Databases []SQLCatalogSpec `json:"databases"`
}
type SQLRegistry struct{ stores map[SQLDatabase]*SQLStore }

type sqlScopeKey struct{}

// LoadSQLRegistry проверяет файлы и их версии до готовности сервиса.
// Пути задаёт оператор в каталоге; клиент передаёт только db_id и db_revision.
func LoadSQLRegistry(ctx context.Context, path string, workers int) (registry *SQLRegistry, err error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, failure("storage_error", "Не удалось открыть каталог SQL-баз.")
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, 1024*1024+1))
	if err != nil || len(data) > 1024*1024 || !validJSONUnicode(data) {
		return nil, failure("invalid_request", "Недопустимый размер или кодировка SQL-каталога.")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var catalog SQLCatalog
	if err = decoder.Decode(&catalog); err != nil {
		return nil, failure("invalid_request", "SQL-каталог не соответствует схеме.")
	}
	var extra any
	if decoder.Decode(&extra) != io.EOF || catalog.Version != 1 || len(catalog.Databases) == 0 || len(catalog.Databases) > 4096 {
		return nil, failure("invalid_request", "Неверная версия или размер SQL-каталога.")
	}
	registry = &SQLRegistry{stores: make(map[SQLDatabase]*SQLStore)}
	metadataBytes := 0
	defer func() {
		if err != nil {
			registry.Close()
		}
	}()
	for _, spec := range catalog.Databases {
		decoded, decodeErr := hex.DecodeString(spec.Revision)
		if !validID(spec.DBID) || spec.Path == "" || decodeErr != nil || len(decoded) != 32 {
			return registry, failure("invalid_request", "Нужны db_id, путь и SHA-256 каждой SQL-базы.")
		}
		key := SQLDatabase{spec.DBID, spec.Revision}
		if _, exists := registry.stores[key]; exists {
			return registry, failure("invalid_request", "Повторная пара db_id/revision в SQL-каталоге.")
		}
		filename := spec.Path
		if !filepath.IsAbs(filename) {
			filename = filepath.Join(filepath.Dir(path), filename)
		}
		store, openErr := OpenSQL(ctx, filename, spec.DBID, workers)
		if openErr != nil {
			return registry, openErr
		}
		if store.info.DBRevision != spec.Revision {
			store.Close()
			return registry, failure("storage_error", "SQL-файл не соответствует ожидаемой ревизии каталога.")
		}
		metadataBytes += len(store.info.Schema)
		if metadataBytes > 64*1024*1024 {
			store.Close()
			return registry, failure("invalid_request", "Суммарные SQL-схемы каталога превышают 64 МиБ.")
		}
		registry.stores[key] = store
	}
	return registry, nil
}
func singleSQLRegistry(store *SQLStore) *SQLRegistry {
	registry := &SQLRegistry{stores: make(map[SQLDatabase]*SQLStore)}
	if store != nil {
		registry.stores[SQLDatabase{store.info.DBID, store.info.DBRevision}] = store
	}
	return registry
}
func (r *SQLRegistry) Close() error {
	var result error
	for _, store := range r.stores {
		if err := store.Close(); err != nil {
			result = err
		}
	}
	return result
}
func (r *SQLRegistry) Databases() []SQLDatabase {
	result := make([]SQLDatabase, 0, len(r.stores))
	for key := range r.stores {
		result = append(result, key)
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].DBID == result[j].DBID {
			return result[i].DBRevision < result[j].DBRevision
		}
		return result[i].DBID < result[j].DBID
	})
	return result
}
func (r *SQLRegistry) Resolve(id, revision string) (*SQLStore, error) {
	if id != "" && revision != "" {
		store := r.stores[SQLDatabase{id, revision}]
		if store == nil {
			return nil, failure("unknown_id", "Неизвестная SQL-база или ревизия.")
		}
		return store, nil
	}
	// Совместимость с одним снимком. При неоднозначности не выбираем «последний».
	var found *SQLStore
	for key, store := range r.stores {
		if (id == "" || key.DBID == id) && (revision == "" || key.DBRevision == revision) {
			if found != nil {
				return nil, failure("invalid_request", "Укажите db_id и db_revision: выбор базы неоднозначен.")
			}
			found = store
		}
	}
	if found == nil {
		return nil, failure("unknown_id", "Неизвестная SQL-база или ревизия.")
	}
	return found, nil
}
