from flask import Flask, jsonify, render_template, request

from app.inference import SingleModelService

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static",
)

service = SingleModelService()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        history = data.get("history") or []

        if not question:
            return jsonify({"error": "Question is required"}), 400

        result = service.ask(question, history)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500