# Оптимизация А/Б тестирования с применением машинного обучени

**Описание**: Этот проект предоставляет сервис, который облегчает проведение и анализ А/Б тестирования с применением методов машинного обучения и предлагает альтернативу АБ-тестам, когда разделение на контроль и тест невозможно. Пользователи могут проводить тесты с синтетическим контролем или сокращать дисперсию текущего теста, используя машинное обучение (CUPED, CUPAC, CUNOPAC), через веб-интерфейс.

## Основные функции

- **Веб-интерфейс**: Пользователи выбирают тип теста, загружают данные и получают результаты.
- **ML Составляющая**: Сервис обучает модели, оптимизирует гиперпараметры и рассчитывает статистические критерии для сокращения дисперсии или создания синтетического контроля.
- **Аналитическая составляющая**: Расчёт и вывод результатов А/Б теста после применения ML.

## Ограничения

- На первой итерации сервис работает с поюзерными тестами с непрерывными целевыми метриками.
- Уменьшение дисперсии зависит от качества данных и предсказательной способности модели.
- Метрика для синтетического контроля должна представлять стационарный временной ряд.

## Этапы разработки

1. Подготовка датасетов и синтетических данных.
2. Автоматизация обучения ML моделей для различных целей.
3. Анализ результатов А/Б тестов.
4. Реализация веб-интерфейса.
5. Интеграция моделей в веб-интерфейс.
6. Визуализация результатов.

## Локальный запуск 
- Установить requirements.txt:  ```pip install -r requirements.txt```
- В директории backend выполнить: ```uvicorn main:app --reload```
- В директории frontend выполнить:  ```streamlit run app.py```
