# Neural Network Architecture

- **Encoder**: `MultiScaleSurfacePointNet2Encoder` (SA1: 1024, SA2: 256, SA3: 64, Global: 1024).
- **Surface Memory**: 320 tokens (256 local SA2 tokens + 64 coarse SA3 tokens).
- **Decoder**: `TargetQueryTransformerDecoder` with 117 learned anatomical query embeddings, 4 layers, 8 heads ($d=256$).
