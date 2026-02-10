# PCB Defect Detection API

FastAPI server for PCB defect detection using CNN model.

## Endpoints
- GET `/` - API status
- POST `/predict` - Upload image for prediction
- GET `/health` - Health check

## Usage
```bash
curl -X POST -F "file=@pcb_image.jpg" https://your-api-url.com/predict
```
