# Cigarette Brand Detection — AI Solution Proposal

**Prepared for:** [Client Name]
**Date:** March 20, 2026
**Prepared by:** [Your Agency Name]

---

## Executive Summary

This proposal outlines three AI-powered solutions for automating cigarette brand detection from field survey photographs. Each option is designed to process images of retail outlet displays, identify which cigarette brands are physically present, and output structured data (brand names, SKUs, and brand counts) — replacing what is currently a manual, time-consuming process.

We have conducted extensive research and testing to identify the approaches that deliver the best balance of **accuracy, cost, and practicality** for the Cambodian market. All three options support the full official brand list (29 mother brands and their SKU variants) and are built to handle real-world conditions: dirty glass, reflections, stacked or rotated packs, and mixed product shelves.

**Our recommendation:** Option 1 (SmartDetect LLM) or Option 2 (CoreVision AI), depending on your expected monthly volume and long-term plans. Details below.

---

## How It Works

Regardless of which option you choose, the workflow is the same:

1. **Upload** your Excel file containing outlet image links
2. The system **fetches each image** and runs AI analysis
3. AI **identifies cigarette brands and SKUs** visible as physical boxes (ignoring shelf labels, signs, and non-cigarette products)
4. **Download** a results Excel file with:
   - Embedded thumbnail images from each outlet
   - Brands detected (Q12A format with Khmer text)
   - SKUs identified (Q12B format)
   - Brand count per outlet
   - Processing status

A web-based dashboard is included with all options for easy, real-time processing.

---

## Option Comparison at a Glance

| | Option 1: SmartDetect LLM | Option 2: CoreVision AI | Option 3: Hybrid Suite |
|---|---|---|---|
| **Price** | **$3,250** | **$4,050** | **$6,800** |
| **Monthly Cost** | $10 - $100 | $50 - $200 | $75 - $300 |
| **Accuracy** | 80 - 90% | 85 - 95% | 90 - 97% |
| **Setup Time** | 1 - 2 weeks | 3 - 5 weeks | 5 - 7 weeks |
| **Training Data Needed** | None | Yes (500+ images) | Yes (500+ images) |
| **Internet Required** | Yes | Yes (server-based) | Yes |
| **Best For** | Fast deployment, lower volumes | High volumes, long-term use | Maximum accuracy requirements |

---

## Option 1: SmartDetect LLM (Third-Party AI)

**Price: $3,250**

### What It Is

Uses a leading third-party AI vision model (Claude or GPT) to analyze each outlet image directly. The AI reads the image, identifies cigarette packs, matches them to the official Cambodian brand list, and returns structured results. No custom training is required — the AI works out of the box.

### What You Get

- Web dashboard for uploading Excel files and viewing results in real time
- Automated image fetching from survey links
- Brand and SKU identification using the official 29-brand list
- Excel output with embedded images, brands, SKUs, and brand counts
- Command-line tool for batch processing large datasets
- Full support for the CHHAT survey data format

### Monthly / Usage Costs

The system charges per image processed through the AI API:

| Monthly Volume | Estimated Monthly Cost |
|---|---|
| 200 images | ~$3 - $5 |
| 500 images | ~$7 - $12 |
| 2,000 images | ~$25 - $40 |
| 5,000 images | ~$65 - $80 |
| 10,000 images | ~$130 - $150 |
| 50,000 images | ~$650 - $700 |

*Hosting cost: $0 - $20/month for the web dashboard.*

### Advantages

- **Fastest to deploy** — operational within 1-2 weeks
- **No training data required** — works immediately with the official brand list
- **High baseline accuracy** (80-90%) out of the box
- **Lowest upfront cost** — $3,250 one-time
- **Cheapest at low volumes** — under 5,000 images/month, this is the most affordable option
- **Easy to update** — adding new brands requires only updating the brand list, no retraining
- **Continuously improving** — as the underlying AI models improve (new releases every few months), your system gets better automatically at no extra cost

### Limitations

- **Per-image cost** — every image processed costs money; at very high volumes (50,000+/month), costs add up
- **Requires internet** — images must be sent to the AI provider's servers for processing
- **Dependent on third-party** — if the AI provider changes pricing or availability, it affects your system
- **Slightly lower accuracy** on heavily obscured or unusual pack orientations compared to a custom-trained model

### Best For

- Teams that want to **start quickly** without a long setup period
- Survey operations processing **under 5,000 images per month**
- Organizations that want the **lowest upfront investment** with predictable per-use costs
- Situations where the **brand list may change frequently** (new brands entering the market)

---

## Option 2: CoreVision AI (Server-Based Custom Model)

**Price: $4,050**

### What It Is

A custom AI model trained specifically on Cambodian cigarette brands using your own survey images. The model is trained to recognize the exact pack designs, colors, and logos used in the Cambodian market. Once trained, it runs on a dedicated server with no per-image API fees.

### What You Get

- Everything in Option 1 (web dashboard, Excel processing, batch tools)
- Custom-trained detection model optimized for Cambodian cigarette brands
- Dedicated server deployment (cloud-hosted)
- Initial model training using your labeled survey images
- One round of model refinement/retraining after initial deployment

### Monthly / Usage Costs

No per-image charges. Fixed monthly hosting cost:

| Hosting Option | Monthly Cost | Best For |
|---|---|---|
| CPU-only server | $20 - $50/month | Under 5,000 images/month |
| GPU-accelerated server | $150 - $300/month | High throughput, 10,000+ images/month |
| Serverless (pay-per-use) | $5 - $30/month | Infrequent / batch processing |

*Optional: Model maintenance and retraining retainer — $150 - $300/month*

### Advantages

- **No per-image fees** — process 100 or 100,000 images at the same cost
- **Higher accuracy** (85-95%) because the model is trained on your actual data
- **Full data control** — images are processed on your own server, not sent to a third party
- **Faster processing** — custom models analyze images in under 100 milliseconds (vs 2-3 seconds for LLM-based)
- **Works in private environments** — can be deployed on-premise if data privacy is a concern
- **Predictable costs** — fixed monthly hosting, no surprises from API price changes
- **Most cost-effective at scale** — at 5,000+ images/month, cheaper than Option 1

### Limitations

- **Longer setup time** (3-5 weeks) — requires collecting and labeling training images
- **Training data required** — needs 500+ labeled images of Cambodian outlet shelves to achieve good accuracy
- **Higher upfront cost** — $4,050 vs $3,250 for Option 1
- **Adding new brands requires retraining** — if a new cigarette brand enters the Cambodian market, the model needs to be updated with new training data (typically 1-2 weeks and additional cost)
- **Fixed hosting cost** — you pay the monthly server cost even during months with low usage

### Best For

- Organizations processing **5,000+ images per month** consistently
- Teams that need **data privacy** (images stay on your own server)
- **Long-term deployments** where predictable, flat monthly costs are preferred
- Situations where **processing speed** matters (batch processing large surveys quickly)

---

## Option 3: Hybrid Intelligence Suite (LLM + Custom AI)

**Price: $6,800**

### What It Is

Combines the best of both approaches: a custom-trained model handles the primary detection, while an LLM-based system acts as a second layer for validation, fallback, and edge cases. When the custom model is uncertain about a detection, it automatically routes the image to the LLM for a second opinion. This dual-layer approach delivers the highest possible accuracy.

### What You Get

- Everything in Option 1 and Option 2
- Dual-layer detection pipeline (custom model + LLM validation)
- Smart routing — only uncertain images go to the LLM (keeping API costs minimal)
- Continuous improvement — disagreements between the two models are flagged for review, creating a feedback loop that improves accuracy over time
- Priority support and quarterly model review

### Monthly / Usage Costs

| Component | Monthly Cost |
|---|---|
| Server hosting (custom model) | $20 - $150 |
| LLM API (fallback only, ~10-20% of images) | $1 - $15 |
| Maintenance and model updates | $150 - $400 |
| **Total** | **$75 - $300/month** |

### Advantages

- **Highest accuracy** (90-97%) through dual-layer verification
- **Smart fallback** — if the custom model is unsure, the LLM catches it; if the LLM is unsure, the custom model provides context
- **Continuous improvement** — the system learns from disagreements between the two models
- **Most resilient** — if one model fails or is unavailable, the other continues working
- **Future-proof** — scalable to detecting other product categories beyond cigarettes

### Limitations

- **Highest upfront cost** — $6,800
- **Most complex system** — more components means more potential points of failure
- **Longest setup time** (5-7 weeks)
- **Requires all dependencies of both Option 1 and 2** — training data, API keys, server hosting

### Best For

- Organizations where **accuracy is the top priority** and errors are costly
- Large-scale operations with **complex detection requirements**
- Teams planning to **expand beyond cigarettes** to other product categories in the future

---

## Our Recommendation

### For most teams: **Start with Option 1, scale to Option 2 when ready**

Here is why:

**Option 1 (SmartDetect LLM at $3,250)** gets you operational in 1-2 weeks with no training data needed. At typical survey volumes (a few hundred to a few thousand images per month), the monthly API costs are minimal — often under $50/month. This lets you:

- Start using the system immediately
- Build confidence in the AI approach with real results
- Collect labeled data naturally (the AI's outputs can be reviewed and corrected, building your training dataset for free)

**Option 2 (CoreVision AI at $4,050)** becomes the better choice once you:

- Process more than ~5,000 images per month consistently
- Have accumulated enough reviewed/corrected data to train a strong custom model
- Need faster processing speeds for large batch jobs
- Want to eliminate per-image API costs entirely

This phased approach means you **don't pay for Option 2's training and setup until you have real data to train on**, and you get value from the system from day one.

---

## Cost Comparison by Volume

To help with budgeting, here is what each option costs per month at different usage levels:

| Monthly Images | Option 1 (monthly) | Option 2 (monthly) | Option 3 (monthly) |
|---|---|---|---|
| 200 | ~$5 | ~$50 | ~$100 |
| 500 | ~$12 | ~$50 | ~$100 |
| 2,000 | ~$40 | ~$50 | ~$110 |
| 5,000 | ~$80 | ~$50 | ~$115 |
| 10,000 | ~$150 | ~$75 | ~$130 |
| 50,000 | ~$700 | ~$100 | ~$150 |

*Option 1 and Option 2 costs cross over at approximately 4,000 - 5,000 images/month. Below that, Option 1 is cheaper. Above that, Option 2 is cheaper.*

---

## What Happens After You Choose

| Step | Option 1 | Option 2 | Option 3 |
|---|---|---|---|
| **Week 1** | System built and deployed | Data collection planning | Data collection planning |
| **Week 2** | Testing and handover | Image labeling begins | Image labeling begins |
| **Week 3** | Live in production | Model training | Model training |
| **Week 4** | — | Testing and refinement | LLM integration |
| **Week 5** | — | Deployment and handover | Testing and refinement |
| **Week 6-7** | — | — | Deployment and handover |

---

## Included With All Options

- Web-based dashboard for real-time processing
- Excel input/output matching your existing survey data format
- Support for the full official Cambodian brand list (29 brands, 100+ SKUs)
- Embedded images in output Excel files
- Batch processing capability for large datasets
- Setup documentation and training session
- 30 days of post-deployment support

---

## Questions?

We are happy to walk through any of these options in detail, provide a live demo, or discuss a custom arrangement that fits your specific needs.

**Contact:** [Your contact details]
