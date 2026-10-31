"use strict";

const $ = (id) => document.getElementById(id);
let settings = null;
let runId = null;
let stream = null;
let busy = false;
let startedAt = 0;
let waitingAt = 0;
let elapsedTimer = null;
let tokens = 0;
let searches = 0;
let lastSequence = 0;
let cancelRequested = false;
const toolRows = new Map();
const modelRows = new Map();

function element(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = String(value);
  return node;
}

function mode() { return document.querySelector('input[name="mode"]:checked').value; }
function seconds(value) { return `${value.toFixed(1)} с`; }
function clock(value) { return `${Math.floor(value / 60).toString().padStart(2, "0")}:${Math.floor(value % 60).toString().padStart(2, "0")}`; }
function formatNumber(value) { return value == null ? "—" : new Intl.NumberFormat("ru-RU").format(value); }

function setStatus(text, kind) {
  $("status").className = `status ${kind}`;
  $("status").replaceChildren(element("span", "dot"), document.createTextNode(text));
}

function setBusy(value) {
  busy = value;
  $("start-button").disabled = value;
  $("stop-button").hidden = !value;
  $("stop-button").disabled = false;
  $("stop-button").replaceChildren(element("span", "", "■"), document.createTextNode("Остановить"));
  for (const input of $("run-form").querySelectorAll("input, textarea")) input.disabled = value;
  if (!value) {
    clearInterval(elapsedTimer);
    $("waiting").hidden = true;
  }
}

function waitFor(label) {
  $("waiting-label").textContent = label;
  $("waiting-time").textContent = "";
  $("waiting").hidden = false;
  waitingAt = Date.now();
}

function row(title, marker, meta = "", kind = "") {
  const wrapper = element("div", `activity-row ${kind}`);
  wrapper.append(element("span", "event-marker", marker));
  const top = element("div", "event-top");
  const heading = element("span", "event-title", title);
  const metadata = element("span", "event-meta", meta);
  top.append(heading, metadata);
  wrapper.append(top);
  $("activity").append(wrapper);
  return { wrapper, heading, metadata };
}

function documentCard(hit, compact = false) {
  const card = element("div", compact ? "corpus-document" : "hit");
  const header = element("div", "hit-header");
  const id = element("span", "document-id");
  id.append(element("i", "document-icon"), document.createTextNode(hit.doc_id));
  header.append(id);
  if (!compact && hit.rank) header.append(element("span", "rank", `#${hit.rank}`));
  card.append(header, element("p", "", hit.snippet || ""));
  return card;
}

function renderProgress(event) {
  const { kind, step, details } = event;
  if (kind === "model_started") {
    $("steps-value").textContent = step;
    modelRows.set(step, row(`Шаг ${step} · запрос к модели`, String(step), `max_tokens ${details.max_output_tokens}`));
    waitFor("Ожидаем ответ модели…");
  } else if (kind === "model_finished") {
    const entry = modelRows.get(step);
    if (entry) {
      entry.heading.textContent = `Шаг ${step} · модель ответила`;
      entry.metadata.textContent = seconds(details.duration_seconds);
      entry.wrapper.append(element("p", "event-description", `${details.model || "Модель"} · ${formatNumber(details.usage.completion_tokens)} выходных токенов`));
    }
    if (details.usage.completion_tokens != null) tokens += details.usage.completion_tokens;
    $("tokens-value").textContent = formatNumber(tokens);
  } else if (kind === "tool_started") {
    const entry = row("Вызов инструмента", "↗", "", "tool");
    entry.heading.replaceChildren(element("span", "tool-name", details.name));
    let argumentsText = details.arguments;
    try {
      const args = JSON.parse(details.arguments);
      argumentsText = details.name === "search" && typeof args?.query === "string" ? args.query : JSON.stringify(args, null, 2);
    } catch { /* Сохраняем исходные аргументы ошибочного вызова. */ }
    entry.wrapper.append(element("code", "tool-arguments", argumentsText));
    const key = `${step}:${details.name}`;
    if (!toolRows.has(key)) toolRows.set(key, []);
    toolRows.get(key).push(entry);
    waitFor(`Выполняем ${details.name}…`);
  } else if (kind === "tool_finished") {
    const key = `${step}:${details.name}`;
    const entry = toolRows.get(key)?.shift() || row(details.name, "↗", "", "tool");
    const result = details.result;
    entry.metadata.textContent = seconds(details.duration_seconds);
    if (result.ok === false) {
      entry.wrapper.classList.add("problem");
      entry.wrapper.append(element("p", "error-message", `${result.error?.code || "tool_error"}: ${result.error?.message || "Ошибка инструмента"}`));
    } else if (details.name === "search") {
      searches += 1;
      $("searches-value").textContent = searches;
      const hits = element("div", "tool-result");
      for (const hit of result.hits || []) hits.append(documentCard(hit));
      if (!result.hits?.length) hits.append(element("p", "event-description", "Поиск не вернул документов."));
      entry.wrapper.append(hits);
    } else {
      entry.wrapper.append(element("p", "tool-finished", "Выбор принят. Эпизод завершён."));
    }
  }
}

function finish(data) {
  const result = data.result;
  setBusy(false);
  const cancelled = result.error?.code === "interrupted";
  if (cancelled) setStatus("Остановлен", "cancelled");
  else if (result.stop_reason === "finished") setStatus("Завершён", "finished");
  else if (result.truncated) setStatus("Бюджет исчерпан", "limited");
  else setStatus("Ошибка эпизода", "error");
  $("steps-value").textContent = result.trajectory.length;
  $("searches-value").textContent = result.search_calls;
  $("tokens-value").textContent = formatNumber(result.usage.completion_tokens);
  $("time-value").textContent = clock(result.duration_seconds);
  $("result-section").hidden = false;
  $("result-heading").textContent = cancelled ? "Эпизод остановлен" : result.truncated ? "Остановка по бюджету" : "Выбранные документы";
  $("result-count").textContent = result.selected_document_ids.length;
  const discovered = new Map((settings?.hits || []).map((hit) => [hit.doc_id, hit]));
  for (const turn of result.trajectory) for (const action of turn.actions) for (const hit of action.result.hits || []) discovered.set(hit.doc_id, hit);
  for (const [index, id] of result.selected_document_ids.entries()) $("result-documents").append(documentCard({ ...discovered.get(id), doc_id: id, rank: index + 1 }));
  if (!result.selected_document_ids.length) $("result-documents").append(element("p", "empty-result", cancelled ? "Частичная траектория сохранена." : "Финальный список пуст. Полученные документы доступны в траектории."));
  if (result.error) {
    $("result-error").textContent = `${result.error.code}: ${result.error.message}`;
    $("result-error").hidden = false;
  }
  $("save-label").textContent = cancelled ? "Частичный результат сохранён" : "Результат сохранён";
  $("saved-path").textContent = data.saved_to;
  $("download").href = data.download_url;
  $("result-footer").hidden = false;
  $("json-section").hidden = false;
  $("result-json").textContent = JSON.stringify(result, null, 2);
  const cost = result.usage.cost;
  $("cost-label").textContent = result.model_metadata.adapter === "mock" ? "Мок · токены условные, стоимость $0" : cost == null ? "Стоимость не указана провайдером" : `Стоимость по usage: $${cost.toFixed(5)}`;
}

function handleEvent(event) {
  if (event.seq <= lastSequence) return;
  lastSequence = event.seq;
  if (event.type === "progress") renderProgress(event.data);
  else if (event.type === "complete") {
    stream.close();
    finish(event.data);
  } else if (event.type === "error" || event.type === "cancelled") {
    stream.close();
    setBusy(false);
    setStatus(event.type === "cancelled" ? "Остановлен" : "Ошибка запуска", event.type === "cancelled" ? "cancelled" : "error");
    row(event.data.message, "!", "", "problem");
  }
}

function reset() {
  if (stream) stream.close();
  runId = null;
  cancelRequested = false;
  lastSequence = 0; tokens = 0; searches = 0;
  toolRows.clear(); modelRows.clear();
  $("activity").replaceChildren();
  $("result-documents").replaceChildren();
  for (const id of ["empty-state", "result-section", "result-footer", "json-section", "form-error", "result-error"]) $(id).hidden = true;
  $("json-section").open = false;
  $("activity").hidden = false;
  $("steps-value").textContent = "0";
  $("searches-value").textContent = "0";
  $("tokens-value").textContent = "—";
  $("time-value").textContent = "00:00";
}

async function jsonRequest(url, body) {
  const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Ошибка запроса: ${response.status}`);
  return data;
}

$("run-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const optionalNumber = (id) => $(id).value.trim() ? Number($(id).value) : null;
  const body = { question: $("question").value, mode: mode(), model_id: $("model-id").value.trim(), max_steps: optionalNumber("max-steps"), max_output_tokens: optionalNumber("max-tokens"), max_tokens_per_step: Number($("per-step").value), request_timeout: Number($("timeout").value) };
  reset(); setBusy(true); setStatus("Выполняется", "running"); waitFor("Запускаем среду…");
  startedAt = Date.now();
  elapsedTimer = setInterval(() => {
    $("time-value").textContent = clock((Date.now() - startedAt) / 1000);
    $("waiting-time").textContent = `${Math.floor((Date.now() - waitingAt) / 1000)} с`;
  }, 500);
  try {
    const data = await jsonRequest("/api/runs", body);
    runId = data.id;
    stream = new EventSource(`/api/runs/${runId}/events`);
    stream.onmessage = (message) => handleEvent(JSON.parse(message.data));
    stream.onerror = () => {
      if (!busy) return;
      if (stream.readyState === EventSource.CLOSED) {
        setBusy(false);
        setStatus("Связь прервана", "error");
        $("form-error").textContent = "Поток событий закрыт. Проверьте сервер UI в терминале.";
        $("form-error").hidden = false;
      } else waitFor("Восстанавливаем поток событий…");
    };
    if (cancelRequested) await jsonRequest(`/api/runs/${runId}/cancel`, {});
  } catch (error) {
    setBusy(false); setStatus("Ошибка запуска", "error");
    $("form-error").textContent = error.message;
    $("form-error").hidden = false;
  }
});

$("stop-button").addEventListener("click", async () => {
  if (!busy) return;
  cancelRequested = true;
  $("stop-button").disabled = true;
  $("stop-button").textContent = "Останавливаем…";
  waitFor("Сохраняем частичную траекторию…");
  if (!runId) return;
  try { await jsonRequest(`/api/runs/${runId}/cancel`, {}); }
  catch (error) {
    $("stop-button").disabled = false;
    $("form-error").textContent = error.message;
    $("form-error").hidden = false;
  }
});

$("run-form").addEventListener("invalid", (event) => {
  const details = event.target.closest("details");
  if (details) details.open = true;
}, true);

function updateMode() {
  const api = mode() === "api";
  $("api-fields").hidden = !api;
  $("mode-hint").textContent = api ? "Модель выбирает запросы и момент завершения. Поиск остаётся моковым." : "Мок выполняет search → finish. Usage условный.";
}
for (const radio of document.querySelectorAll('input[name="mode"]')) radio.addEventListener("change", updateMode);
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && !busy) $("run-form").requestSubmit();
});

async function initialize() {
  try {
    const response = await fetch("/api/settings");
    if (!response.ok) throw new Error("Не удалось загрузить настройки UI.");
    settings = await response.json();
    $("model-id").value = settings.model_id;
    $("corpus-count").textContent = settings.hits.length;
    for (const hit of settings.hits) $("corpus-documents").append(documentCard(hit, true));
    if (settings.api_error) {
      $("api-note").textContent = settings.api_error;
      $("api-note").hidden = false;
    }
  } catch (error) {
    $("form-error").textContent = error.message;
    $("form-error").hidden = false;
  }
}
initialize();
