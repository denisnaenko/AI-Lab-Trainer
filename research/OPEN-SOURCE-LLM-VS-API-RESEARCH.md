# RESEARCH: Open-source LLM vs. API (Anthropic, OpenAI) for AI Lab Trainer

*Date: 2026-09-29. Model names, prices and licenses change quickly, so treat every
concrete version and number below as a starting point to re-check before you commit.*

---

## 1. TL;DR

- **Weights matter less than the validation loop.** The pipeline already checks every
  generated lab mechanically: the reference solution must pass, the starter must fail,
  and auto-checks must cover at least 80% of requirements. Whichever model you pick, that
  loop is what makes the output trustworthy. It also gives you a free **reward signal**
  for fine-tuning later (see 5.C).
- **Short term (MVP, acceptance criteria):** keep a frontier model behind the existing
  `LLMClient` interface, and add an **OpenAI-compatible client** so the same code can run
  against a self-hosted open model (vLLM or Ollama), a Russian provider, or Claude/OpenAI.
  That code change is about 40 lines.
- **Medium term:** store every teacher-approved lab and every teacher edit as a dataset.
  Fine-tuning only pays off once you have a few hundred approved labs. Before that,
  few-shot examples pulled from the lab library give most of the same benefit.
- **Long term (the research contribution):** fine-tune an open model (Qwen3-class,
  7–32B) with **SFT, then RL with a verifiable reward** from the validation loop, and
  measure it against the API baseline on a fixed benchmark. That comparison can serve as a
  thesis chapter or paper in its own right.
- **The Russian context raises the priority of open models.** Anthropic and OpenAI do not
  officially serve Russia (access and payment are both a problem), universities often
  require on-prem hosting, and students doing the labs need an LLM they can actually reach.
  So the product should be **provider-agnostic from day one**, even if Claude is the
  strongest generator.

---

## 2. What this task demands from a model

| Requirement | Why it matters here | Who handles it better |
|---|---|---|
| Long, schema-valid structured output (many files in one JSON) | `messages.parse` into Pydantic (`LabPlan`, `LabAssignment`) | API has native structured outputs. Open models get the same through constrained decoding in vLLM/SGLang (xgrammar, outlines). |
| Correct, runnable Python code, with tests consistent with the solution | The validation loop rejects bad labs, and each repair costs another call | Frontier API has the edge. The best open coder models (Qwen3-Coder, DeepSeek, GLM, Kimi K2) come close; small (≤14B) models are clearly weaker. |
| Knowledge of *current* agent frameworks (tool calling, MCP, LangGraph, Agents SDKs) | Labs about agentic development go stale quickly | API models are refreshed by the vendor. A fine-tuned open model freezes at its training cutoff unless you add RAG over docs. |
| Good pedagogical Russian | Brief and instructions are in Russian | Both are fine at the top end. Small open models degrade in Russian; Russian-tuned models help (T-pro, GigaChat, Vikhr). |
| Consistent house style, fixed difficulty ladder, course conventions | Teachers want predictable output | **This is where fine-tuning wins**, though few-shot prompting also gets far. |
| Low volume | Tens to hundreds of labs per semester, not millions | Makes the API very cheap in absolute terms (see section 4). |

The implication: generation quality is capped by **code reasoning**, which favours large
models. Style and format, the parts fine-tuning improves, are secondary, and prompting
plus examples already covers them.

---

## 3. Two uses of an LLM in this project (don't mix them up)

1. **Generator LLM**: the one inside `lab_trainer` that writes labs. Only the teacher's
   side uses it, at low volume.
2. **Student-runtime LLM**: the one the *student's agent* calls while doing the lab
   (`llm.complete(messages, tools=...)` in `labs/tool-agent-basic/starter/agent.py`).
   Tests use `FakeLLM`, but students will want to run their agent for real.

The choice for (2) is arguably *more* constrained than for (1). Russian students
generally can't pay for Claude or OpenAI. The labs should therefore target an
**OpenAI-compatible endpoint**, with a documented local option (Ollama plus a small
tool-calling model such as Qwen3-8B or gpt-oss-20b) and optional university-hosted vLLM,
GigaChat or YandexGPT. That is a strong, practical reason to support open models even if
the generator stays on an API.

---

## 4. Cost and infrastructure, rough numbers

**One generated lab**: 3 stages plus up to 3 repair rounds is roughly 50–150k tokens,
mostly output (code). At frontier API prices, that is **on the order of $0.5–3 per lab**.
Even 500 labs a year costs tens to low hundreds of dollars.

**Self-hosting** (order-of-magnitude figures):

| Model size | Inference hardware | Fine-tuning hardware (LoRA/QLoRA) | Notes |
|---|---|---|---|
| 7–9B dense | 1× 24 GB GPU (RTX 4090/3090, L4) | 1× 24 GB with QLoRA | Fine for the student runtime. Weak as a lab generator. |
| 14B dense | 1× 24–48 GB (4-bit on 24 GB) | 1× 48 GB | Reasonable generator after fine-tuning. |
| 30–32B dense / ~30B MoE | 1× 80 GB (A100/H100) or 2× 48 GB; 4-bit fits in 24–48 GB | 1× 80 GB QLoRA | **Sweet spot** for an on-prem generator. |
| 100B+ MoE (DeepSeek-V3.x, Qwen3-235B, GLM-4.5, Kimi K2, gpt-oss-120b) | multi-GPU node (gpt-oss-120b fits one 80 GB GPU) | Hard for a university lab. Use the API versions of these instead. | Closest to frontier quality. |

A rented H100 costs roughly $2–4/hour. **At this project's volume, the API is cheaper
than dedicated hardware.** Self-hosting makes economic sense only if (a) the university
already has GPUs, (b) data must stay on-prem, or (c) you serve the student runtime to
many students.

---

## 5. Variants

### A. Pure API (current: Claude via `messages.parse`)
- **Pros:** best code quality today, native structured outputs, no infrastructure,
  models refreshed for you, and the fastest route to the acceptance criteria.
- **Cons:** no fine-tuning of Claude by you (OpenAI offers fine-tuning of some models via
  its API, but the result stays a closed model on their servers). Vendor lock-in,
  availability and payment from Russia, and data leaving the country.
- **Best for:** the MVP, and as a quality ceiling or baseline for research.

### B. Open-weights model, prompted, not fine-tuned
- **Candidates:**
  - General and code: Qwen3 / Qwen3-Coder, DeepSeek-V3.x, GLM-4.5/4.6, Kimi K2,
    gpt-oss-20b/120b, Mistral (Small/Devstral), Llama.
  - Russian-focused: T-pro 2.0 (T-Bank, Qwen3-based), GigaChat open-weight releases
    (Sber), Vikhr models.
  - *Verify current versions and licenses: Apache-2.0 and MIT are easy, while Llama and
    some others have custom licenses.*
- **Serving:** vLLM or SGLang (OpenAI-compatible API, constrained JSON-schema decoding,
  tool calling). Ollama or llama.cpp for laptops and students.
- **Pros:** runs on-prem, no per-token cost, works in Russia, one API shape for both
  generator and student runtime.
- **Cons:** the ops burden is yours, and first-try validation pass rates will be lower,
  so expect more repair rounds.
- **Best for:** an on-prem deployment option, the student runtime, and a baseline before
  fine-tuning.

### C. Open model + fine-tuning (your main interest)
Stages, in increasing value and effort:
1. **SFT (LoRA/QLoRA)** on pairs of *(topic, difficulty, teacher notes) → approved lab
   JSON*. This teaches format, style, the difficulty ladder and course conventions.
   It needs roughly 300–1000+ high-quality examples.
2. **Preference tuning (DPO/ORPO)** on *(generated draft, teacher-edited final)* pairs.
   The teacher's edit becomes a "chosen vs. rejected" signal, so the model learns what
   teachers actually fix.
3. **RL with verifiable rewards (GRPO-style).** This is the interesting part. The
   validation loop *is* a reward function:
   - reward += solution passes all tests
   - reward += starter fails the auto-checked tests (tests actually test something)
   - reward += auto-check coverage ≥ 80%
   - reward += schema-valid, lint-clean, runtime under a limit
   - optional: an LLM judge scores the pedagogical quality of brief and instructions
   This is the same recipe used to train code and reasoning models. The task fits it
   unusually well because correctness is machine-checkable.
- **Tooling:** Unsloth (cheapest single-GPU LoRA), Hugging Face TRL (SFT/DPO/GRPO),
  Axolotl, LLaMA-Factory, and verl or OpenRLHF for larger RL.
- **Pros:** a model specialised to your task, which can let a 14–32B model match much
  larger general models *on this narrow task*. Real research novelty, and full control.
- **Cons:** you need data first, sandboxed test execution at scale for RL, evaluation
  infrastructure, and upkeep. Knowledge of new frameworks freezes, so you need RAG or
  periodic retraining.

### D. Distillation: API as teacher, open model as student
Generate thousands of labs with the frontier API, keep only those that **pass
validation** (and ideally a sample reviewed by teachers), then SFT an open model on them.
This solves the cold-start data problem for C.
- ⚠️ **Check the terms of service.** Anthropic's and OpenAI's terms restrict using outputs
  to build competing models. Read the current terms for your case, or use an open
  teacher model with a permissive license (Qwen, DeepSeek, GLM, gpt-oss) to generate
  the synthetic data.

### E. Hybrid router
- Use the frontier model for the hard stages (plan, and solution plus tests) and an
  open or small model for cheap stages (Russian polishing, rubric, repair of trivial
  failures).
- Or run open/fine-tuned as the primary and fall back to the API when validation fails
  after N attempts.
- Every call is logged, so the data for C accumulates automatically.

### F. Russian hosted APIs (GigaChat, YandexGPT / Yandex AI Studio)
- Legally and financially accessible in Russia, with data kept in Russia, which fits
  152-ФЗ and university procurement. Gitverse is Sber's platform, so a GigaChat pairing
  is natural for that export path.
- Yandex Cloud also hosts some open models, and GigaChat and YandexGPT offer
  fine-tuning services. Check what is currently available.
- Code quality on this task is unknown until measured, so include them in the benchmark
  (section 7).

---

## 6. Comparison

| | A. API | B. Open, prompted | C. Open + FT | D. Distill | E. Hybrid | F. RU API |
|---|---|---|---|---|---|---|
| Quality today | ★★★★★ | ★★★ | ★★★–★★★★ (after work) | ★★★★ | ★★★★★ | ★★★? (measure) |
| Fine-tunable | ✗ (OpenAI: partly) | – | ✓ | ✓ | partly | partly (vendor FT) |
| Works from Russia | ✗ / hard | ✓ | ✓ | ✓ (after training) | partly | ✓ |
| On-prem / data control | ✗ | ✓ | ✓ | ✓ | partly | in-country |
| Infra effort | none | medium | high | high | medium | none |
| Cost at your volume | very low | GPU-bound | GPU-bound plus training | one-off API spend | low | low |
| Research value | baseline | baseline | **high** | high | medium | medium |

---

## 7. Evaluation plan (makes the choice empirical)

Build a **fixed benchmark** once and run every candidate through it:
- About 20 topics × 3 difficulty levels = 60 generation requests (e.g. tool calling,
  memory, RAG agent, multi-agent, MCP server, guardrails, evaluation of agents, ...).
- **Metrics:**
  1. First-try validation pass rate
  2. Mean repair rounds
  3. Auto-check coverage
  4. Share of labs where tests really discriminate (starter fails, solution passes;
     optionally mutation testing of the solution)
  5. Schema errors
  6. Teacher edit distance on a sample
  7. Blind teacher rating 1–5 on pedagogy
  8. Cost and latency per lab
- **Candidates:** Claude (current), one OpenAI model, Qwen3-32B / Qwen3-Coder,
  gpt-oss-120b, DeepSeek or GLM via API, GigaChat, YandexGPT, then your fine-tuned
  model(s).

The grading and validation code already in the repo computes metrics 1–5 almost for
free.

---

## 8. What this means for the code (small, low-risk)

`generation/llm.py` already defines an `LLMClient` protocol with one method,
`structured(system, prompt, schema)`. Suggested additions, **not done yet**:
- `OpenAICompatibleClient(base_url, model, api_key)`: uses
  `response_format={"type": "json_schema", ...}` from the Pydantic schema. It covers vLLM,
  SGLang, Ollama, OpenRouter, OpenAI, and the providers exposing OpenAI-compatible
  endpoints.
- Config: `LAB_TRAINER_PROVIDER=anthropic|openai_compat`, plus `LAB_TRAINER_BASE_URL`
  and `LAB_TRAINER_MODEL`.
- A **generation log** (`runs/*.jsonl`): request, each stage's prompt and output,
  validation result, and the final teacher-approved diff. This becomes the SFT/DPO/RL
  dataset for variant C.
- Student labs: document running the agent against any OpenAI-compatible endpoint,
  with Ollama as the default local option.

---

## 9. Recommended roadmap

| Phase | What | Outcome |
|---|---|---|
| 1 (now) | Keep Claude as the generator. Add the OpenAI-compatible client and the generation log. Finish the 5 labs and meet the acceptance criteria. | Working product, provider-agnostic |
| 2 | Build the benchmark (section 7). Run API vs. open prompted vs. RU APIs. Add few-shot retrieval from approved labs. | Data-driven choice of default; first research results |
| 3 | Collect approved labs and teacher edits (or distill, variant D, after checking terms). SFT a Qwen3-class 14–32B model with LoRA. | First fine-tuned model, compared on the benchmark |
| 4 | DPO on teacher edits, then GRPO with the validation-loop reward. | The research contribution: a small, on-prem model specialised for lab generation |
| 5 | Deploy fine-tuned model on university GPUs, with API fallback (variant E). | On-prem production setup |

---

## 10. Risks and open questions

- **Stale knowledge:** agentic frameworks change monthly, and a fine-tuned model won't
  know about new ones. Mitigate with RAG over framework docs in the prompt.
- **Reward hacking in RL:** a model can learn trivial tests that pass and fail
  "correctly" without testing anything real. Mitigate with mutation testing, minimum
  test counts per requirement, and an LLM or teacher spot-check.
- **Sandboxing:** RL means running thousands of generated test suites. You need
  isolated containers with no network access and time limits.
- **Licenses:** check model licenses (Llama and some others carry restrictions) and API
  terms for synthetic data.
- **Data and privacy:** generating labs involves no personal data, but *grading student
  submissions with an LLM* would. Keep grading deterministic (as now), or keep it
  on-prem to stay clear of 152-ФЗ issues.
- **Evaluation of pedagogy:** automated metrics can't tell whether a lab teaches well.
  Budget teacher time for blind ratings.
