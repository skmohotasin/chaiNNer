# Quality Bulk 1024 — chaiNNer chain

File: `quality_bulk_1024.chn`

## What it does
1. Loads **all images** from an input folder (bulk)
2. Improves quality with AI models
3. If width **&lt; 1024px** → 4x UltraSharp upscale, then ensures width ≥ 1024
4. If width **≥ 1024px** → 1x StarSample enhance (quality, no forced size jump)
5. Saves results to an output folder

## How to use
1. Open chaiNNer
2. File → Open → `D:\GitHub\chaiNNer\chains\quality_bulk_1024.chn`
3. Put images in `D:\GitHub\chaiNNer\chains\bulk_input`  
   (or change the folder on **Bulk Load Images**)
4. Optional: change **Save Bulk Output** folder (default `bulk_output`)
5. Press **Run**

## Models used
- Small images: `models\4x-UltraSharpV2.safetensors`
- Large images: `models\1x_StarSample_V2.0_NS.safetensors`

You can swap either Load Model path to another model you prefer.
