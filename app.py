from flask import Flask
from blueprints.ui import ui_bp
from blueprints.projects import projects_bp
from blueprints.training import training_bp
import os
import secrets

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    secrets.token_hex(32)
)


app.register_blueprint(ui_bp)
app.register_blueprint(projects_bp)
app.register_blueprint(training_bp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(
        host="127.0.0.1",
        port=port,
        debug=False,
    )
