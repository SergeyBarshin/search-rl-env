# Источники реального примера

- Задача `train_a2s_4`: [Snowflake/HyDRA-Bench](https://huggingface.co/datasets/Snowflake/HyDRA-Bench),
  ревизия `176151eea2e171b674af961fa5d1b2b33eaa6e76`, `train_a2s.json`.
  Авторы: Ruofan Wu, Boyi Liu, Fan Shu, Yite Wang, Zhewei Yao, Yuxiong He, Feng Yan.
  Карточка указывает CC-BY-SA-4.0. Публичный вопрос сохранён в task.json,
  данные проверки — отдельно в reference.json.
- Тексты: [CoRAG KILT corpus](https://huggingface.co/datasets/corag/kilt-corpus),
  ревизия `bf95d9b4b19c68420395efad7498d0ef5052ab4b`, шард 15 из 37,
  Parquet row group 435. Это конвертация Wikipedia/KILT, а не текст эталонного ответа.
  Из 1000 записей исключена одна с пустым содержимым. Остальные 999 текстов
  перенесены без переписывания, исходный passage id сохранён как doc_id.
  Изменения: переименование полей и добавление language=en.
- Авторы исходных статей — участники Wikipedia. Ссылка для каждого документа:
  `https://en.wikipedia.org/?curid=WIKIPEDIA_ID`, где id берётся из
  `provenance.json → corpus.row_to_wikipedia_id`; история правок доступна со страницы
  статьи. KILT использует снимок 2019-08-01. У исходных текстов Wikipedia сохраняются
  условия атрибуции/share-alike ([условия Wikimedia](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)).
  Карточка конвертации CoRAG помечена apache-2.0; эта отметка не заменяет условия
  исходных текстов Wikipedia.
- SQL-файл `hockey.sqlite` в Git не включён. Это BIRD train database в
  [публикации IBM VAKRA](https://huggingface.co/datasets/ibm-research/VAKRA),
  ревизия `1388b9f1aaed887a73957eba651a6c2034010476`.
  Карточка публикации указывает CC-BY-NC-SA-4.0; исходные материалы BIRD имеют
  собственные условия. Пример предназначен для локальной исследовательской проверки.
- Опциональный энкодер: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
  ревизия `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, Apache-2.0.
  Веса не включены в Git. Используется опубликованный ONNX int8-файл и исходный tokenizer.json.

Точные SHA-256, идентификаторы и правила отбора находятся в provenance.json.
Ручные reference ids обозначают фрагменты страницы Toronto Arenas; это не официальные qrels HyDRA.
