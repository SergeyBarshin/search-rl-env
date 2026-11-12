// Команда сборки корпуса и запуска отдельного поискового сервиса.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/SergeyBarshin/search-rl-env/services/search/internal/search"
)

func run(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return &search.Error{Code: "invalid_request", Message: "Используйте build, serve или serve-sql."}
	}
	switch args[0] {
	case "build":
		flags := flag.NewFlagSet("build", flag.ContinueOnError)
		input := flags.String("input", "examples/fixtures/hydra/corpus.jsonl", "JSONL с документами")
		output := flags.String("output", "indexes/hydra.sqlite", "Файл нового снимка")
		corpus := flags.String("corpus-id", "hydra-a2s4-kilt-pilot", "Id корпуса")
		revision := flags.String("revision", "", "Ревизия источника; по умолчанию хеш")
		chunkChars := flags.Int("chunk-chars", 2000, "Размер фрагмента в символах")
		if err := flags.Parse(args[1:]); err != nil {
			return err
		}
		if flags.NArg() != 0 {
			return &search.Error{Code: "invalid_request", Message: "Неожиданные аргументы."}
		}
		file, err := os.Open(*input)
		if err != nil {
			return &search.Error{Code: "invalid_request", Message: "Не удалось открыть входной JSONL."}
		}
		defer file.Close()
		manifest, err := search.Build(ctx, file, *output, *corpus, *revision, *chunkChars)
		if err != nil {
			return err
		}
		return json.NewEncoder(os.Stdout).Encode(manifest)
	case "serve", "serve-sql":
		flags := flag.NewFlagSet(args[0], flag.ContinueOnError)
		sqlOnly := flags.Bool("sql-only", args[0] == "serve-sql", "Запуск только SQL без поискового индекса")
		catalog := flags.String("sql-catalog", "", "JSON-каталог исходных SQL-баз и ревизий")
		snapshot := flags.String("snapshot", "indexes/hydra.sqlite", "Готовый снимок")
		listen := flags.String("listen", "127.0.0.1:8090", "HTTP-адрес")
		sqlSnapshot := flags.String("sql-snapshot", "", "Необязательная исходная SQLite-база для SQL-тулы")
		dbID := flags.String("db-id", "", "Id исходной SQL-базы")
		config := search.DefaultConfig()
		flags.IntVar(&config.Workers, "workers", config.Workers, "Одновременные операции с SQLite")
		flags.IntVar(&config.MaxPending, "max-pending", config.MaxPending, "Выполняемые и ожидающие операции суммарно")
		flags.DurationVar(&config.Timeout, "timeout", config.Timeout, "Deadline операции с очередью")
		flags.IntVar(&config.MaxResults, "max-results", config.MaxResults, "Максимум результатов")
		flags.IntVar(&config.SnippetChars, "snippet-chars", config.SnippetChars, "Максимум символов сниппета")
		flags.IntVar(&config.MaxReadChars, "max-read-chars", config.MaxReadChars, "Максимум символов чтения")
		flags.IntVar(&config.MaxQueryChars, "max-query-chars", config.MaxQueryChars, "Максимум символов запроса")
		flags.IntVar(&config.MaxQueryTerms, "max-query-terms", config.MaxQueryTerms, "Максимум терминов запроса")
		if err := flags.Parse(args[1:]); err != nil {
			return err
		}
		if flags.NArg() != 0 {
			return &search.Error{Code: "invalid_request", Message: "Неожиданные аргументы."}
		}
		logger := slog.New(slog.NewJSONHandler(os.Stderr, nil))
		if *catalog != "" && (*sqlSnapshot != "" || *dbID != "") {
			return &search.Error{Code: "invalid_request", Message: "Выберите sql-catalog или пару sql-snapshot/db-id."}
		}
		if *sqlOnly && *catalog == "" && *sqlSnapshot == "" {
			return &search.Error{Code: "invalid_request", Message: "SQL-only режим требует sql-catalog или sql-snapshot/db-id."}
		}
		var service *search.Service
		var err error
		if *sqlOnly {
			service, err = search.NewSQLService(config, logger)
		} else {
			service, err = search.Open(ctx, *snapshot, config, logger)
		}
		if err != nil {
			return err
		}
		defer service.Close()
		var handler http.Handler = service.Handler()
		databaseCount := 0
		if *catalog != "" {
			registry, err := search.LoadSQLRegistry(ctx, *catalog, config.Workers)
			if err != nil {
				return err
			}
			defer registry.Close()
			databaseCount = len(registry.Databases())
			handler = service.HandlerWithRegistry(registry)
		} else if *sqlSnapshot != "" || *dbID != "" {
			if *sqlSnapshot == "" || *dbID == "" {
				return &search.Error{Code: "invalid_request", Message: "Нужны оба флага sql-snapshot и db-id."}
			}
			store, err := search.OpenSQL(ctx, *sqlSnapshot, *dbID, config.Workers)
			if err != nil {
				return err
			}
			defer store.Close()
			databaseCount = 1
			handler = service.HandlerWithSQL(store)
		}
		listener, err := net.Listen("tcp", *listen)
		if err != nil {
			return &search.Error{Code: "storage_error", Message: "Не удалось открыть HTTP-порт."}
		}
		defer listener.Close()
		server := &http.Server{Handler: handler, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: config.Timeout, WriteTimeout: config.Timeout + time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 16 * 1024,
			BaseContext: func(net.Listener) context.Context { return ctx }}
		if *sqlOnly {
			logger.Info("sql_ready", "address", listener.Addr().String(), "databases", databaseCount, "config", config)
		} else {
			logger.Info("search_ready", "address", listener.Addr().String(), "manifest", service.Info(), "databases", databaseCount)
		}
		done := make(chan error, 1)
		go func() { done <- server.Serve(listener) }()
		select {
		case err := <-done:
			if !errors.Is(err, http.ErrServerClosed) {
				return err
			}
		case <-ctx.Done():
			shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			if err := server.Shutdown(shutdown); err != nil {
				server.Close()
			}
			<-done
		}
		return nil
	default:
		return &search.Error{Code: "invalid_request", Message: "Неизвестная команда; используйте build, serve или serve-sql."}
	}
}
func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := run(ctx, os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, search.SafeError(err))
		os.Exit(1)
	}
}
