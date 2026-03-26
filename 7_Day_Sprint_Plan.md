# CHHAT AI — 7-Day Sprint Plan

## Custom Model: Cigarette Brand Detection → 90% Accuracy

---

## Objective

Deliver the first Excel file with cigarette brand/SKU detection results using a fully custom model (no third-party API in production). Target: **90%+ brand detection accuracy within 7 days**.

---

## Team Roster (5 People)

| Role | Responsibility |
| --- | --- |
| **Sprint Lead** | Overall coordination, model testing, quality decisions, video recording |
| **Annotator A** | Image annotation on Roboflow (bounding boxes around cigarette packs) |
| **Annotator B** | Image annotation on Roboflow (bounding boxes around cigarette packs) |
| **Data Collector** | Photograph/film all cigarette brands, process media, assist with labeling |
| **ML Engineer** | GPU setup (RunPod), model training, deployment, Excel generation |

---

## Daily Breakdown

### Day 1 — Data Collection Blitz

> **Goal:** Collect as much raw training data as possible in parallel.

| Role | Task | Target Output |
| --- | --- | --- |
| Sprint Lead | Record 15-second video of each cigarette pack (front side + health warning side, slow 360° rotation) | 15+ brands × 2 sides = 30 videos |
| Annotator A | Annotate survey images on Roboflow — draw bounding boxes around every cigarette pack | 30 images annotated |
| Annotator B | Annotate survey images on Roboflow — draw bounding boxes around every cigarette pack | 30 images annotated |
| Data Collector | Photograph each brand pack: front, back, top, angled, dim lighting, with other products nearby | 15 brands × 6 shots = 90 photos |
| ML Engineer | Set up RunPod A100 GPU account, clone repository, install dependencies, run test training | RunPod environment ready |

**Priority brands to collect (Tier 1 — must have by end of Day 1):**
- MEVIUS, ESSE, FINE, LUXURY, IZA, GOLD SEAL, 555, ARA, CAMBO

**Tier 2 (important):**
- WINSTON, COW BOY, LAPIN, ORIS, COCO PALM, CROWN

---

### Day 2 — Auto-Label + First Training Run

> **Goal:** Process collected data, merge all sources, run first model training on GPU.

| Role | Task | Target Output |
| --- | --- | --- |
| Sprint Lead | Process pack videos into training frames (video_processor.py), review auto-labeled images | 2,000+ training images from videos |
| Annotator A + B | Continue Roboflow annotation — target 100 total annotated images | 100 images with bounding boxes |
| Data Collector | Use Gemini auto-labeler to generate bounding boxes + brand labels on all 93 survey images, then review for obvious errors | 93 images auto-labeled |
| ML Engineer | Merge ALL training data, kick off RF-DETR training + brand classifier training on RunPod A100 | Model v1 trained (~2 hours) |

**End of Day 2 deliverable:** First trained model with real-world data.

---

### Day 3 — Test + Identify Errors

> **Goal:** Find every mistake the model makes. Document error patterns.

| Role | Task | Target Output |
| --- | --- | --- |
| Sprint Lead | Test Model v1 on 20+ held-out images, document every wrong detection (missed pack, wrong brand, false positive) | Error report with specific images and brands |
| Annotator A + B | Annotate the specific images/brands the model got WRONG — focus entirely on error cases | 50 error-case annotations |
| Data Collector | Record additional videos of the brands that performed worst in testing | 10 additional videos |
| ML Engineer | Analyze confusion matrix (which brands get confused with each other), prepare optimized retraining dataset | Retraining dataset ready |

**Error documentation format:**

| Image | Expected | Model Said | Error Type |
| --- | --- | --- | --- |
| shelf_001.jpg | WINSTON | MEVIUS | Brand confusion (both blue) |
| shelf_002.jpg | IZA (3 packs) | IZA (1 pack) | Missed detections |
| shelf_003.jpg | — | COW BOY | False positive (health warning image) |

---

### Day 4 — Retrain v2

> **Goal:** Train an improved model using error corrections from Day 3.

| Role | Task | Target Output |
| --- | --- | --- |
| Sprint Lead | Test Model v2, compare accuracy against v1 side-by-side | Accuracy comparison report |
| Annotator A + B | Final annotation pass — any remaining unannotated images | All available images annotated |
| Data Collector | Label brand names on detected crops from Model v1 output | 200+ labeled crops |
| ML Engineer | Train Model v2 on RunPod with expanded error-corrected dataset | Model v2 trained |

---

### Day 5 — Retrain v3 + Validate

> **Goal:** Final training iteration. Validate 90%+ accuracy.

| Role | Task | Target Output |
| --- | --- | --- |
| Sprint Lead | Test Model v3, verify 90%+ brand accuracy on holdout set | Accuracy report: pass/fail |
| All Annotators | Final error corrections and edge case fixes if accuracy < 90% | Final corrections |
| ML Engineer | Train Model v3 if needed, optimize model for inference speed | Final model ready |

**Accuracy validation protocol:**
1. Hold out 20 images that the model has NEVER seen during training
2. Run model on all 20
3. Count: correct brands / total brands = accuracy %
4. If < 90%: identify top 3 error patterns → annotate more of those → retrain

---

### Day 6 — Generate Output

> **Goal:** Process the client's full dataset and produce the delivery Excel files.

| Role | Task | Target Output |
| --- | --- | --- |
| Sprint Lead | Run full client dataset through final custom model | Raw results Excel |
| Annotator A + B | Manual review of every row in the output — flag errors | Reviewed + corrected Excel |
| ML Engineer | Generate both export formats: Detailed Report + Client Format (Q12A/Q12B) | Final Excel files |

---

### Day 7 — Deliver + Buffer

> **Goal:** Final quality check and delivery.

| Role | Task | Target Output |
| --- | --- | --- |
| Everyone | Final quality check on exported Excel files | ✅ |
| Sprint Lead | Send deliverables to client | **DELIVERED** |
| ML Engineer | Document model version, training data stats, known limitations | Handoff notes |

---

## Training Data Targets

| Data Source | Expected Images | Responsible | Due |
| --- | --- | --- | --- |
| Survey images — Roboflow bounding box annotations | 93 | Annotator A + B | Day 2 |
| Pack video recordings → extracted frames | 2,000+ | Sprint Lead + Data Collector | Day 2 |
| Pack photographs (6 angles per brand) | 90+ | Data Collector | Day 1 |
| Gemini auto-labeled survey crops | 335 | Data Collector (review) | Day 2 |
| Error-case re-annotations | 50–100 | Annotator A + B | Day 3–4 |
| Existing Roboflow datasets (already downloaded) | 654 | Pre-existing | Done |
| Synthetic data (generated from reference images) | 500 | Automated | Done |
| **Total training data** | **~3,500+** | | |

---

## GPU Training Setup (ML Engineer — Day 1)

### RunPod A100 Setup

```
1. Sign up at runpod.io — add $25 credit (enough for 5 training runs)
2. Create Pod → Select A100 80GB → PyTorch template
3. Clone repository:
   git clone https://github.com/richardnhek/chhat-ai-agent.git
   cd chhat-ai-agent
   pip install -r requirements.txt
   pip install rfdetr pytorch-lightning pycocotools faster-coco-eval easyocr

4. Upload training data via scp or rsync

5. Train RF-DETR (pack detection):
   python train_local.py --epochs 50 --batch-size 8

6. Train brand classifier:
   python brand_classifier.py --epochs 30 --batch-size 64
```

### Training cost per run

| Component | GPU Time | Cost |
| --- | --- | --- |
| RF-DETR (50 epochs) | ~1.5 hours | ~$3.75 |
| Brand Classifier (30 epochs) | ~15 minutes | ~$0.60 |
| **Total per cycle** | | **~$4.35** |
| **Budget for 5 cycles** | | **~$22** |

---

## Roboflow Annotation Guide (Annotators A + B)

### Setup

1. Log in to [app.roboflow.com](https://app.roboflow.com)
2. Open the project (or create: "CHHAT Cigarette Detection" → Object Detection)
3. Upload images from `survey_images/` folder

### Annotation Rules

- Draw a tight bounding box around **every visible cigarette pack or carton**
- Use a single class: `cigarette_pack`
- **DO** annotate: individual packs, cartons, sleeves, partially visible packs
- **DO** annotate packs that are upside down, sideways, behind glass
- **DO NOT** annotate: brand logos on shelf labels, signs, or posters
- **DO NOT** annotate: non-cigarette products (lighters, candy, drinks)
- If a pack is less than ~15 pixels wide, skip it (too small to be useful)

### Export

- Format: **COCO JSON**
- Download zip → place in `training_data/roboflow_annotated/`

---

## Video Recording Guide (Sprint Lead + Data Collector)

### Equipment

- Smartphone camera (matches field surveyor quality)
- Plain surface (white paper or dark table)

### For each cigarette brand

1. Place pack on surface — **front side up**
2. Record 15 seconds: slowly rotate 360° (front → side → back → side → front)
3. Flip to **health warning side** — record another 10 seconds same rotation
4. Move camera closer and farther to vary the scale
5. **Do NOT use flash**

### File naming

Name each video with the brand:

```
mevius_original_front.mp4
mevius_original_warning.mp4
esse_change_front.mp4
iza_ff_front.mp4
```

### Processing

```
python video_processor.py --dir videos/ --retrain
```

This extracts frames, applies augmentation (rotation, blur, brightness), and retrains the brand classifier automatically.

---

## Critical Path

```
Day 1: Collect (annotations + videos + photos + GPU setup)     ← ALL PARALLEL
Day 2: Merge + Train v1 on A100                                ← FIRST MODEL
Day 3: Test v1 → error report → targeted re-annotation         ← FIND MISTAKES
Day 4: Retrain v2 with corrections                             ← FIX MISTAKES
Day 5: Test v2 → final fixes → v3 if needed                    ← VALIDATE 90%
Day 6: Generate Excel from final model                          ← OUTPUT
Day 7: Deliver + buffer                                         ← SHIP
```

---

## Why This Reaches 90%

| Factor | Impact |
| --- | --- |
| 3,500+ real-world training images (vs 168 studio photos currently) | Fixes the domain gap — the #1 accuracy blocker |
| 3 train-test-fix cycles with error correction between each | Each cycle targets the specific mistakes from the previous one |
| A100 GPU enables 1.5-hour training (vs 45 hours on local Mac) | Allows rapid iteration — 3 cycles in 3 days |
| Focused on 15 relevant brands (removing 14 irrelevant ones) | Less confusion between classes = higher per-brand accuracy |
| 5 people × 7 days = 35 person-days of effort | Sufficient human effort to generate quality training data |
| Health warning sides included in training | Eliminates the biggest source of false positives |

---

## Success Criteria

| Metric | Target | How to Measure |
| --- | --- | --- |
| Brand detection accuracy | ≥ 90% | Correct brands / total brands on 20 holdout images |
| Pack detection recall | ≥ 85% | Found packs / actual packs on holdout images |
| False positive rate | ≤ 10% | Wrong detections / total detections |
| Processing speed | ≤ 5 min for 28 outlets | Timed end-to-end run |
| Excel output matches client format | Q12A + Q12B columns correct | Manual comparison with sample |

---

## Risks

| Risk | Mitigation |
| --- | --- |
| Not enough annotated images by Day 2 | Use Gemini auto-labeler to supplement (training tool only, not production) |
| Model accuracy plateaus below 90% | Focus annotations on the specific brands/conditions that fail |
| RunPod GPU unavailable | Fallback: Google Colab A100 (free tier) or Lambda Labs |
| Team unfamiliar with Roboflow | Sprint Lead demos annotation workflow in first 30 minutes of Day 1 |
| Client data has brands not in training set | Flag as "UNKNOWN" — better to admit uncertainty than guess wrong |
