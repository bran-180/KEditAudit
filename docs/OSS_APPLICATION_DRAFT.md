# Codex for Open Source application draft

This is a working draft. Replace projections with repository evidence before submission. The official program currently targets core maintainers and widely used public projects, while allowing applicants to explain ecosystem importance when a project does not fit the usual criteria: [Codex for Open Source](https://developers.openai.com/community/codex-for-oss).

## Project name

**KEditAudit**. The name and distribution namespace were checked on 2026-08-16;
availability must be checked again immediately before public publication.

## Project summary — English

KEditAudit is an open-source, editor-agnostic audit toolkit for measuring the intended and unintended effects of knowledge edits in autoregressive language models. It compares baseline and edited model states across efficacy, paraphrase generality, neighborhood locality, logical portability, control-distribution drift, and structural weight changes, then emits a versioned JSON report with complete experiment provenance. Rather than duplicating existing editing frameworks, KEditAudit provides a reproducible evidence layer that can consume outputs from tools such as the official ROME implementation and EasyEdit.

## 專案摘要 — 繁體中文

KEditAudit 是一套開源、與編輯演算法解耦的知識編輯審計工具，用於量測自回歸語言模型在編輯前後的預期效果與非預期影響。它比較 baseline 與 edited model 在編輯效能、釋義泛化、鄰域保真、邏輯可遷移性、控制分布漂移與權重結構變化上的差異，並輸出包含完整實驗來源的版本化 JSON 報告。KEditAudit 不重複打造另一套編輯算法集合，而是提供可接收官方 ROME、EasyEdit 等工具輸出的可重現證據層。

## Community impact — English

Knowledge-editing research has many algorithms and benchmarks, but results remain difficult to compare when model revisions, tokenizers, prompt sets, seeds, aggregation rules, and baseline states are not recorded consistently. KEditAudit aims to give researchers, students, and open-source model maintainers a small common artifact format and an offline reproducible test path. The project will publish metric definitions, raw per-probe evidence, limitations, and adapters under a permissive open-source license.

## 社群價值 — 繁體中文

知識編輯領域已具有多種演算法與 benchmark，但只要模型版本、tokenizer、prompt 集合、隨機種子、彙整方法或 baseline 狀態沒有一致記錄，結果便難以比較與重現。KEditAudit 希望為研究者、學生與開源模型維護者提供一套精簡的共通 artifact 格式，以及不依賴 GPU 或網路的基本測試路徑。專案將以寬鬆的開源授權公開指標定義、逐筆證據、限制與外部框架 adapter。

## Why Codex and OpenAI support — English

Codex would support the maintainer workflow rather than serve as an unverified scientific oracle. We plan to use it to implement issue-scoped changes, expand deterministic unit and integration tests across supported model adapters, review benchmark-result diffs in pull requests, keep citations and documentation aligned with code, and triage reproducibility failures reported by contributors. Optional OpenAI API use would be limited to generating candidate probes for human review and summarizing already-computed structured results; the core metrics will remain local, deterministic, and usable without an API key.

## 為什麼需要 Codex 與 OpenAI 支援 — 繁體中文

Codex 在此專案中將支援維護流程，而不是作為未經驗證的科學判定者。我們計畫用它完成 issue 範圍明確的修改、擴充各模型 adapter 的確定性單元與整合測試、審查 pull request 中的 benchmark 差異、保持引用文件與程式碼一致，並協助處理社群回報的重現性問題。OpenAI API 僅作為選配，用於生成待人工審核的 probe 候選或摘要既有結構化結果；核心指標將維持本地、可重現且不需要 API key。

## Resource plan

Do not promise a fixed token budget before measurement. Report planned usage in auditable work units:

1. maintainer development: issue implementation, tests, documentation, and PR review;
2. compatibility work: pinned adapters and reproducibility investigations;
3. optional dataset assistance: candidate paraphrases or portability probes that receive human review and provenance labels;
4. release operations: benchmark-diff summaries and migration notes.

For each category, record the number of runs, model used, approximate tokens/cost where available, and the public issue or release it supported.

## Evidence checklist before applying

- public repository and complete license;
- at least one tagged release;
- CI and offline unit tests;
- one reproducible tiny-model audit;
- one external editor adapter;
- public issues/PRs showing actual maintenance;
- honest list of supported and unsupported models;
- at least one external reproduction or user report;
- measured description of how Codex/API resources improve maintenance.

## Claims to avoid

- “guarantees safety”;
- “detects all backdoors”;
- “supports all Transformers”;
- projected users, stars, benchmarks, or token volume presented as actual evidence;
- claiming that OpenAI API use is required when the core project works locally.
