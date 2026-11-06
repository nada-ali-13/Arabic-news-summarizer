# Qualitative evaluation sheet

Status: **pending actual trained-model outputs and human review**. Use 10–20 saved examples from `best_test.json`, include a range of source lengths and topics, and keep the IDs. Do not choose only good examples. Compare the same rows against `baseline_test.json`.

| ID | Factual claims supported? | Essential information retained? | Repetition? | Complete ending? | Arabic fluency 1–5 | Output length suitable? | Notes |
|---|---|---|---|---|---|---|---|
| pending | | | | | | | |

Read the source for each factual claim; a reference summary can omit valid facts. Flag invented people, dates, counts and causation explicitly. Inspect word counts and unfinished endings. Note whether source truncation omitted relevant text. The UI's multi-window strategy requires separate long-document review and is not covered by truncated-input Trainer scores.
