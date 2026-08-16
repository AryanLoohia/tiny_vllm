import uvicorn
import sys
import os

if __name__ == "__main__":
    print("====================================================")
    print(" Starting tiny-vLLM Educational Engine...")
    print(" API:       http://127.0.0.1:8000")
    print(" Dashboard: http://127.0.0.1:8000/")
    print("====================================================")
    uvicorn.run("server.api:app", host="127.0.0.1", port=8000, reload=False)
