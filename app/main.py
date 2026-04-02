# app/main.py

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

def health_check():
    return JSONResponse(content={'status': 'ok', 'version': '1.0.0'}, status_code=status.HTTP_200_OK)

@app.get('/health')
def read_health():
    return health_check()
