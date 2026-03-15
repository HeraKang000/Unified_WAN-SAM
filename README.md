# VaceSamhera

ComfyUI pipeline for AI-powered commercial video production.  
Combines text-prompted segmentation (SAM3) with video generation (Wan2.1 VACE / Wan2.2 Animate) to enable identity-preserving clothing/face swaps and reference-driven animation — fully automated from a single person photo.

---

## What it does

| Mode | Input | Output |
|------|-------|--------|
| **MV2V** — Masked Video-to-Video | Person photo + reference image | Video with swapped clothing or face region |
| **R2V** — Reference-to-Video | Reference image | New animated video from still |
| **WanAnimate** | Person photo + pose/depth guide | Motion-controlled animation (Wan2.2) |

---

## Pipeline

```
Person photo
    └─▶ SAM3Grounding (text prompt: "shirt", "face")
            └─▶ Mask
                    └─▶ VACE MV2V (ref image conditioning)
                                └─▶ Video output
                                        └─▶ MMAudio (synced audio)
```

---

## Custom Nodes (SAMhera)

Built on top of Meta's SAM3. Extended with a stateless architecture for reliable video tracking and text-prompted detection.

| Node | Description |
|------|-------------|
| `LoadSAM3Model` | Loads SAM3 with ComfyUI memory management. Auto-downloads from HuggingFace if not found. |
| `SAM3Grounding` | Text-prompted object detection — finds regions matching a description (e.g. `"person"`, `"red shirt"`). |
| `SAM3Segmentation` | Click/point-based segmentation. |
| `SAM3MultipromptSegmentation` | Multi-region segmentation from multiple prompts simultaneously. |
| `SAM3VideoSegmentation` | Initializes stateless video tracking session with point/box prompts on first frame. |
| `SAM3Propagate` | Propagates masks across all video frames. State flows through outputs — no global mutable state. |
| `SAM3VideoOutput` | Renders final per-frame masks as ComfyUI image batch. |

**Design highlights:**
- Stateless video tracking — inference state is reconstructed on demand, no session cleanup needed
- Unified model loads once for both image and video workflows
- Auto-download from HuggingFace on first use
- GPU/CPU offloading via ComfyUI `model_management`

---

## Models

| Model | Size | Purpose |
|-------|------|---------|
| Wan2.1 T2V 14B (fp8) | ~15 GB | VACE base diffusion model |
| Wan2.1 VACE module (fp8) | ~3 GB | Masked video conditioning |
| Wan2.2 Animate 14B (fp8) | ~15 GB | Pose/depth-guided animation |
| Wan VAE | 508 MB | Video encode/decode |
| UMT5-XXL (fp8 + fp16) | — | Text encoder |
| SigCLIP vision | — | Reference image conditioning (R2V/MV2V) |
| CLIP vision H | — | Image conditioning (WanAnimate) |
| SAM3 | — | Segmentation backbone |
| CausVid LoRA v1/v2 | 319/205 MB | Fast inference acceleration |
| WanAnimate Relight LoRA | — | Scene relighting (LightX2V) |

All fp8 models run on 16 GB VRAM (RTX 4080 SUPER) via `blocks_to_swap`.

---

## Setup

Single-command provisioning — installs ComfyUI, all custom nodes, models, and dependencies:

```bash
curl -fsSL https://raw.githubusercontent.com/HeraKang000/VaceSamhera/main/provisioning_unified.sh | bash
```

Or use `entrypoint_unified.sh` as the instance startup script for cloud GPU environments.

**On-demand package install after boot:**
```bash
source /workspace/install_pkg.sh
ipkg <package> [version_spec]
```

---

## Stack

- **ComfyUI** — workflow runtime
- **SAM3** (Meta) — segmentation backbone
- **Wan2.1 VACE / Wan2.2 Animate** (Wan-AI / Kijai fp8 repacks) — video generation
- **CausVid LoRA** — fast inference
- **MMAudio** — synchronized audio synthesis
- **Python** / **PyTorch** / **ONNX Runtime**

---

## Hardware

Developed and tested on RTX 4080 SUPER (16 GB VRAM, 64 GB RAM, Windows).  
fp8 quantization brings the 14B model from ~28 GB to ~14 GB VRAM footprint.
