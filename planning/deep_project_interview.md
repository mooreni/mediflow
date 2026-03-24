# deep_project_interview.md - MediFlow Interview Transcript

## Interview Summary

**Date:** 2026-03-24
**Requirements file:** planning/requirements.md

---

## Q&A Transcript

**Q: Is Milestone 2 intentionally absent from the requirements?**
A: Yes — intentional gap. The project has exactly four milestones: 1, 3, 4, 5.

**Q: Is any code written yet, or is this greenfield?**
A: Greenfield — starting from scratch, no existing code.

**Q: How should the work be split for /deep-plan?**
A: One spec per milestone (4 specs total), mirroring the milestone structure exactly.

**Q: Which part feels most technically uncertain/risky?**
A: DSPy optimizer tuning — getting per-agent DSPy optimization to actually improve translation quality.

**Q: Does the Gold Standard evaluation dataset exist or need to be created?**
A: Already exists — there's an existing annotated dataset to use.

**Q: What is the target output language?**
A: Russian. The pipeline translates Hebrew medical documents → Russian.

---

## Key Decisions and Constraints

- **Language pair:** Hebrew → Russian
- **Document types:** ER summaries, doctor summaries, prescriptions (5 categories: Summary, Prescript, Referral, Form, Record)
- **Models:** Gemini Pro for translation/evaluation, Gemini Flash for routing
- **Framework:** DSPy for prompt optimization
- **Evaluation dataset:** Already exists (gold standard Hebrew + Russian reference translations)
- **Drug names:** Must NOT be translated in prescriptions — left in English/original form
- **OCR target:** Hebrew text extraction from scanned medical documents (Google Cloud Vision API)
- **Primary technical risk:** DSPy optimizer tuning quality

## Split Structure Decision

4 splits, one per milestone, in sequential dependency order:
1. `01-evaluation` — Evaluation Mechanism (Milestone 1)
2. `02-agent-orchestration` — Agent Orchestration & Classification (Milestone 3)
3. `03-xai-transparency` — Transparency / XAI (Milestone 4)
4. `04-ocr-ui` — OCR & UI Integration (Milestone 5)
