#!/bin/bash
# Startup script executed by Lambda Web Adapter.
# LWA sets PORT; default to 8080 if not set.
exec uvicorn handler:app --host 0.0.0.0 --port "${PORT:-8080}"
