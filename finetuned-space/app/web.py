import os

from app.api import app
from configs.config import TrainConfig

cfg = TrainConfig()

if __name__ == "__main__":
    port = int(os.getenv("PORT", cfg.flask_port))
    app.run(host=cfg.flask_host, port=port, debug=False)