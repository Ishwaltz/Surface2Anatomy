---
title: Surface2Anatomy Live Inference API
emoji: 🫀
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# Surface2Anatomy 3D Organ Location Prediction API

Deep Neural Network backend for target-conditioned 3D internal anatomy localization from external body surface geometry.

## API Endpoints
- **GET** `/` - Service info
- **GET** `/health` - Health status, PyTorch device & model status
- **GET** `/targets` - 121 anatomical targets catalog
- **POST** `/predict` - Single target 3D localization
- **POST** `/predict-multiple` - Multi-target localization in a single pass
- **GET** `/docs` - Interactive OpenAPI / Swagger interface
