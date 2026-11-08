package search

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
)

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(value)
}
func writeError(w http.ResponseWriter, err error) {
	safe := SafeError(err)
	status := http.StatusInternalServerError
	switch safe.Code {
	case "invalid_request":
		status = http.StatusBadRequest
	case "unknown_id":
		status = http.StatusNotFound
	case "busy":
		status = http.StatusServiceUnavailable
	case "timeout":
		status = http.StatusGatewayTimeout
	case "cancelled":
		status = http.StatusRequestTimeout
	}
	writeJSON(w, status, map[string]any{"error": safe})
}
func decodeRequest(w http.ResponseWriter, r *http.Request, destination any) error {
	r.Body = http.MaxBytesReader(w, r.Body, 64*1024)
	data, err := io.ReadAll(r.Body)
	if err != nil {
		return failure("invalid_request", "Превышен размер запроса или ошибка чтения.")
	}
	if !validJSONUnicode(data) {
		return failure("invalid_request", "Неверный UTF-8.")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err = decoder.Decode(destination); err != nil {
		return failure("invalid_request", "JSON не соответствует схеме.")
	}
	var extra any
	if decoder.Decode(&extra) != io.EOF {
		return failure("invalid_request", "Ожидался один JSON-объект.")
	}
	return nil
}

// Handler не зависит от адреса listener; тестируется через httptest.
func (s *Service) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		if !s.Healthy() {
			writeError(w, failure("storage_error", "Снимок недоступен."))
			return
		}
		writeJSON(w, 200, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /v1/manifest", func(w http.ResponseWriter, r *http.Request) {
		if !s.Healthy() {
			writeError(w, failure("storage_error", "Снимок недоступен."))
			return
		}
		writeJSON(w, 200, s.Info())
	})
	mux.HandleFunc("POST /v1/search", func(w http.ResponseWriter, r *http.Request) {
		var request SearchRequest
		if err := decodeRequest(w, r, &request); err != nil {
			writeError(w, err)
			return
		}
		result, err := s.Search(r.Context(), request)
		if err != nil {
			writeError(w, err)
			return
		}
		writeJSON(w, 200, result)
	})
	mux.HandleFunc("POST /v1/read", func(w http.ResponseWriter, r *http.Request) {
		var request ReadRequest
		if err := decodeRequest(w, r, &request); err != nil {
			writeError(w, err)
			return
		}
		result, err := s.Read(r.Context(), request)
		if err != nil {
			writeError(w, err)
			return
		}
		writeJSON(w, 200, result)
	})
	mux.HandleFunc("POST /v1/chunks", func(w http.ResponseWriter, r *http.Request) {
		var request ChunksRequest
		if err := decodeRequest(w, r, &request); err != nil {
			writeError(w, err)
			return
		}
		result, err := s.Chunks(r.Context(), request)
		if err != nil {
			writeError(w, err)
			return
		}
		writeJSON(w, 200, result)
	})
	return mux
}
