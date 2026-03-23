# Training Strategy — How to Reach 97-98% Accuracy

## Research Sources
- Internal analysis (Claude Opus 4.6)
- External consultation (GPT-4o)
- Enterprise spec documents (RealCHHATProject/)
- Roboflow, Google Vertex AI, academic papers

---

## The 4-Layer Training Data Strategy

### Layer 1: Brand-Agnostic Detection (available NOW)

Train RF-DETR to detect "cigarette_pack" as a single class using existing datasets:

| Dataset | Images | Source |
|---|---|---|
| Cigarette Packs by Florin | 479 | Roboflow |
| Cigarette Box by daay | varies | Roboflow |
| Cigarette Detection by IRC | varies | Roboflow |
| Cigarette Pack Detection by myplay | varies | Roboflow |
| SKU-110K (retail shelf) | 9,998 | Ultralytics |

**Expected mAP@50:** 70-80% on generic cigarette pack detection.
**Timeline:** 1-2 days (download + train).

### Layer 2: Synthetic Data Augmentation (1 week)

Use the 234 reference images from the brand book to generate 5,000-10,000 synthetic training images:

**Pipeline:**
1. Take a clean reference pack image
2. Apply augmentations: rotation (0-360°), perspective warp, brightness/contrast variation, blur, glass reflection overlay, partial occlusion
3. Paste onto shelf background images at random positions
4. Generate COCO-format bounding box annotations automatically

**Libraries:** OpenCV, PIL, imgaug, albumentations
**Expected mAP boost:** +10-20% over Layer 1 alone.

### Layer 3: Auto-Labeled Real Images (ongoing)

Use Gemini to auto-generate bounding box annotations on the client's real survey images:
1. Send image to Gemini: "Draw bounding boxes around every cigarette pack, return coordinates"
2. Human verifies/corrects in the annotation tool
3. Each verified image becomes training data

**Expected data volume:** 500+ images within first month of client usage.

### Layer 4: Active Learning Corrections (continuous)

Every correction from the visual annotation tool becomes training data:
1. User draws rectangle on missed pack → COCO annotation
2. User selects brand + SKU → classification label
3. Accumulated corrections trigger retraining (threshold: 200+ new annotations)

---

## Recognition Strategy: Three Signals

### Signal 1: EasyOCR (fast, free, runs on crops only)
- Reads text on cropped pack images
- Fuzzy matches against brand/SKU list
- **Reliability:** 60-70% on clean text, drops to 30-40% on dirty/angled/Khmer text
- **Critical rule:** OCR runs ONLY on RF-DETR crops, NOT on full images (avoids shelf label confusion)
- **Use as:** Supplementary signal, not primary

### Signal 2: Visual Embedding Search (recommended primary)
- Embed each cropped pack using CLIP/DINOv2
- Compare against 234 reference image embeddings
- Top-k similarity match determines brand
- **Expected top-1 accuracy:** 80-90% on clean crops
- **Model recommendation:** DINOv2 (same backbone as RF-DETR, consistent features)
- **Vector store:** numpy dot product for 234 images (FAISS overkill at this scale)
- **Handles rotation/occlusion:** Apply test-time augmentation (4 rotations + original, take best match)

### Signal 3: Gemini Arbitrator (expensive, high accuracy, hard cases only)
- Only invoked when Signal 1 + Signal 2 disagree or both are low confidence
- Expected invocation rate: 10-20% of crops
- **Cost control:** Budget-capped, rate-limited

### Fusion Logic
```
IF ocr_confidence > 0.85 AND embedding_top1_similarity > 0.80:
    → Use OCR result (both agree, skip Gemini)
ELIF embedding_top1_similarity > 0.85:
    → Use embedding result (visual match is strong)
ELIF ocr_confidence > 0.75:
    → Use OCR result (text is readable)
ELSE:
    → Escalate to Gemini arbitrator
```

---

## RF-DETR Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Model | RF-DETR Base | DINOv2 backbone |
| Epochs | 100-150 | More epochs for small datasets |
| Learning rate | 1e-4, cosine decay | Start small for fine-tuning |
| Batch size | 8-16 | Depends on GPU memory |
| Classes | 1 initially ("cigarette_pack") | Add brand classes in Phase 2 |
| Backbone freeze | First 20 epochs | Unfreeze after warmup |
| Image size | 560px | RF-DETR default |
| Augmentations | Mosaic, rotation, blur, perspective, color jitter | Match real-world conditions |

**Expected mAP@50:**
- Layer 1 only (Western data): 70-80%
- + Layer 2 (synthetic): 80-88%
- + Layer 3 (auto-labeled real): 85-92%
- + Layer 4 (corrections): 90-95%

---

## Accuracy Progression Timeline

| Week | Data | Brand F1 | SKU F1 | How |
|---|---|---|---|---|
| 1-2 | Western datasets + synthetic | 0.82-0.88 | 0.70-0.78 | RF-DETR + OCR + Gemini arbitrator |
| 3-4 | + 200 auto-labeled real images | 0.88-0.92 | 0.78-0.85 | + Visual embedding search |
| 5-8 | + 500 corrections | 0.92-0.95 | 0.85-0.90 | First retraining cycle |
| 9-16 | + 1,000 corrections | 0.95-0.97 | 0.88-0.93 | Second retraining |
| 17+ | + ongoing corrections | 0.97-0.98 | 0.90-0.95 | Continuous improvement |

---

## Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Synthetic data doesn't represent real conditions | Mix with auto-labeled real images, validate on holdout set |
| OCR reads shelf labels instead of pack text | OCR only runs on RF-DETR crops, never full images |
| New SKU/packaging change | Reference catalog update + retraining trigger |
| Model drift over time | Monthly holdout evaluation, drift monitoring |
| Client annotation quality | Annotation guidelines + validation checks |
