"""云端启动入口：读取环境变量 PORT，以 0.0.0.0 启动 Streamlit"""
import os
import sys
from streamlit.web import cli as stcli

if __name__ == "__main__":
    port = os.getenv("PORT", "8501")
    sys.argv = [
        "streamlit", "run", "app.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--browser.gatherUsageStats", "false",
    ]
    sys.exit(stcli.main())
