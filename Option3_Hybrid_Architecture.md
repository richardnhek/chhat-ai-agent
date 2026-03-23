# Option 3: Hybrid Intelligence Suite — Enterprise Architecture

## Source Documents
- `RealCHHATProject/engineering-blueprint.rtf`
- `RealCHHATProject/full-critical-technical-specs.rtf`
- `RealCHHATProject/handoff-ready-technical-specification.rtf`

---

## Architecture: Multi-Stage CV-First Pipeline

The system uses a **detect → crop → recognize** pipeline, NOT a "send full image to LLM" approach.

### Pipeline Flow
```
Image Ingest → Image QA → Pack Detector (RF-DETR) → Crop Normalization
→ OCR (PaddleOCR) + Visual Retrieval + Hierarchical Classifier
→ VLM Arbitrator (Gemini, hard cases ONLY)
→ Fusion & Calibration → Auto-approve or Human Review
→ Active Learning → Scheduled Retraining
```

### Six Non-Negotiable Design Rules
1. Detector is NOT the SKU recognizer — it finds packs (class-agnostic or coarse-brand)
2. Recognition happens on CROPS, not full shelves
3. OCR is a CORE signal, not optional — fine-grained SKU variants differ by tiny wording
4. Retrieval is MANDATORY — canonical SKU catalog with top-k similarity search
5. VLM (Gemini) is the ARBITRATOR, not the main worker — hard cases only
6. Human review is a PERMANENT subsystem — this is what makes 97-98% achievable

---

## Model Selection

| Role | Model | License |
|------|-------|---------|
| Detector | RF-DETR | Apache 2.0 |
| OCR | PaddleOCR | Apache 2.0 |
| Retrieval encoder | Fine-tuned SigLIP-based encoder | Self-hosted |
| Classifier | Hierarchical (Head A: brand, Head B: SKU) | Custom |
| VLM Arbitrator | Gemini 2.5 Flash (hard cases only) | Vertex AI |
| Vector store | Postgres + pgvector → FAISS later | Open source |

---

## Accuracy Targets (Phased)

| Phase | Mother-Brand F1 | SKU F1 | Review Rate |
|-------|----------------|--------|-------------|
| Phase 1 | 0.90-0.94 | 0.78-0.86 | 20-30% |
| Phase 2 | 0.94-0.96 | 0.85-0.91 | 12-20% |
| Phase 3 | 0.97-0.98 | 0.90-0.95 | <12% |

**Key insight:** "97-98% is a system outcome, not a model choice."

---

## Training Data Requirements

### Three Dataset Layers
- **Layer A (Detection):** COCO-format bounding boxes, attributes: occluded, rotated, reflection_heavy, stacked
- **Layer B (Crop Recognition):** Crops with mother_brand, exact_sku, packaging_generation, OCR reference text
- **Layer C (Arbitrator):** JSONL image examples for Gemini supervised fine-tuning (100-500 examples)

### Data Split Policy
Split by OUTLET, survey wave, region, device — NOT random by image (prevents leakage)

### Labeling Policy
- Annotators must use `unknown_brand`, `unknown_sku` — NEVER guess unreadable text
- Use Label Studio for annotation + active learning
- Use FiftyOne for detection evaluation and failure analysis

---

## Retraining Triggers
- 200+ reviewed corrections since last cycle
- OR 30+ corrections in one SKU family
- OR unknown rate spike
- OR packaging refresh confirmed
- OR drift alert from monitoring

---

## Infrastructure (Google Cloud)

| Component | Technology |
|-----------|-----------|
| Compute | Cloud Run (CPU for API/OCR; GPU for detector) |
| Queue | Cloud Tasks |
| Storage | Cloud Storage with signed URLs |
| Database | Postgres + pgvector |
| MLOps | Vertex AI Pipelines + Experiments |
| VLM | Vertex AI (Gemini 2.5 Flash) |
| Tuning | Vertex AI supervised fine-tuning |
| Annotation | Label Studio + FiftyOne |
| Frontend | Next.js |

---

## Cost Control
- Gemini batch inference at 50% discount for large surveys
- VLM arbitrator is BUDGET-CAPPED and rate-limited
- Image QA gate rejects bad images before expensive processing
- Track: cost per 1,000 images

---

## Scaling to New Categories
- Hierarchical taxonomy supports new levels
- Retrieval-based recognition: adding new SKU = adding catalog entries, not retraining
- Class-agnostic detector doesn't need retraining for new SKUs
- Packaging lifecycle control with active date windows
