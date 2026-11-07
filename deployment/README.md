---
title: Arabic News Summarizer
emoji: 📰
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
python_version: '3.12'
app_file: app.py
pinned: false
---

Optional future hosting template for **خُلاصة** by **ندى علي**. Nothing is deployed.
The trained weights are in a private Google Drive folder and are not bundled or
publicly downloadable through GitHub. Hosting requires separately permitted access
to the complete model export; a temporary Colab `gradio.live` share is not permanent hosting.

Copy this front matter to the Space's root README only when preparing an authorized deployment.
Set `PYTHONPATH=src` and `ARABIC_SUMMARIZER_MODEL` to a complete exported model/tokenizer directory.
Provide sufficient measured RAM/VRAM. A trained model is not bundled with this source.
Resolve upstream weight licensing before public redistribution; XL-Sum is CC BY-NC-SA 4.0.
