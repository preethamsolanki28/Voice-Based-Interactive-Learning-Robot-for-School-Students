"""
Voice Chatbot — Flask Backend
-------------------------------
Handles one route for the SPA and one route for LLM chat via OpenRouter.
All speech-to-text and text-to-speech happen entirely in the browser
(Web Speech API + SpeechSynthesis), so this server stays lean.

Run:
    python app.py
    then open http://localhost:5000
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import logging

# ── Third-party ───────────────────────────────────────────────────────────────
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# ── Load environment variables from .env before anything else ─────────────────
# python-dotenv reads OPENROUTER_API_KEY (and anything else in .env)
# into os.environ so the rest of the app can use os.getenv().
load_dotenv()

# ── Flask app initialization ──────────────────────────────────────────────────
app = Flask(__name__)

# ── Logging setup ─────────────────────────────────────────────────────────────
# INFO level gives us useful runtime info without debug noise.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# OpenRouter's OpenAI-compatible chat completions endpoint.
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model: GPT-4o Mini — extremely cheap (~$0.00005/request) and highly capable.
# Understands speech transcription artifacts far better than free models.
# Swap MODEL_ID here to change models globally.
MODEL_ID = "openai/gpt-4o-mini"

# Hard ceiling on how long we wait for the LLM (seconds).
# The client also has a 35-second timeout as a safety net.
REQUEST_TIMEOUT_SECONDS = 30

# Max characters we accept from the browser per message.
# Prevents abuse and keeps costs predictable.
MAX_MESSAGE_LENGTH = 2000

# ── System prompt ─────────────────────────────────────────────────────────────
# Injected before every user message.
# Two key concerns:
#   A) TTS output format — no markdown, short sentences, plain English.
#   B) STT input noise — browser speech recognition regularly mishears
#      math symbols, technical terms, and proper nouns. The prompt tells
#      the model to infer intent from context rather than take garbled
#      words literally.
SYSTEM_PROMPT = (
    "You are a professional voice assistant. "
    "The user is speaking to you; their words are transcribed by a browser speech recognition engine "
    "which frequently mishears or phonetically mangles words — especially for math, science, and technical terms. "
    "Your job is to infer the user's true intent from context, even when the transcription is imperfect.\n\n"

    "UNDERSTANDING VOICE TRANSCRIPTION ARTIFACTS:\n"
    "- 'hall square', 'whole square', 'whole squared' = whole squared, as in (A+B)²\n"
    "- 'a plus b whole square' = (A+B)², which expands to A² + 2AB + B²\n"
    "- 'pie' or 'pi' = π (the mathematical constant)\n"
    "- 'root' or 'square root of X' = √X\n"
    "- 'X squared', 'X to the power 2' = X²\n"
    "- 'X cubed', 'X to the power 3' = X³\n"
    "- Letters like 'a', 'b', 'x', 'y', 'n' followed by math words are mathematical variables\n"
    "- 'sigma' = Σ (summation), 'delta' = Δ, 'theta' = θ, 'lambda' = λ\n"
    "- 'integral of', 'differentiate', 'derivative of' = calculus operations\n"
    "- Garbled proper nouns, company names, or technical terms: infer from context\n"
    "- If a word sounds like it might be a mishearing of a technical term, treat it as such\n\n"

    "OUTPUT RULES:\n"
    "1. Be concise — 1 to 3 sentences max unless the topic genuinely requires more detail.\n"
    "2. No markdown, no bullet points, no asterisks, no code fences.\n"
    "3. Do NOT repeat or echo the user's garbled transcription. Answer the inferred question directly.\n"
    "4. Never open with filler words like 'Certainly', 'Of course', 'Sure', or 'Great question'. "
    "Begin your answer immediately.\n"
    "5. FORMULAS AND MATH: always write them using proper Unicode symbols — ², ³, √, π, Σ, ∫, ±, ≈, ≠, ≤, ≥, ∞, θ, λ, Δ, α, β, γ, etc. "
    "For example: write '(A+B)² = A² + 2AB + B²' not 'A plus B whole squared equals A squared plus 2AB plus B squared'. "
    "The app will convert symbols to spoken words before reading them aloud — you do not need to spell them out.\n"
    "6. Prose surrounding formulas should still be plain conversational English.\n"
    "7. Spell out non-math abbreviations: 'as soon as possible' not 'ASAP'.\n"
    "8. Match the user's register: casual if casual, technical if technical.\n"
    "9. If you cannot answer, say so in one sentence.\n"
    "10. Never fabricate facts. If uncertain, say you are not sure."
)


# ── Route: Home ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the single-page application."""
    return render_template("index.html")


# ── Route: Health check ───────────────────────────────────────────────────────
@app.route("/health")
def health():
    """
    Lightweight liveness + configuration check.
    Returns whether the API key is set — never its value.
    Useful for debugging deployment issues.
    """
    return jsonify({
        "status": "ok",
        "model": MODEL_ID,
        "api_key_configured": bool(os.getenv("OPENROUTER_API_KEY")),
    }), 200


# ── Route: Chat ───────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    """
    Receive a transcribed user message, send it to OpenRouter,
    and return the LLM's reply as JSON.

    Request body (JSON):
        { "message": "the user's spoken text" }

    Response (success):
        { "reply": "the assistant's response" }

    Response (error):
        { "error": "human-readable description" }
    """

    # ── Guard 1: API key must be configured ──────────────────────────────────
    # Read on every request so a hot-reload of .env works without restart.
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY is not set in the environment.")
        return jsonify({"error": "Server configuration error: API key is missing."}), 500

    # ── Guard 2: Body must be valid JSON ─────────────────────────────────────
    # silent=True makes get_json() return None instead of raising on bad input.
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    # ── Guard 3: Message must be a non-empty string ───────────────────────────
    user_message = data.get("message", "")
    if not isinstance(user_message, str):
        return jsonify({"error": "The 'message' field must be a string."}), 400
    user_message = user_message.strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    # ── Guard 4: Enforce length limit ─────────────────────────────────────────
    if len(user_message) > MAX_MESSAGE_LENGTH:
        return jsonify({
            "error": f"Message exceeds the {MAX_MESSAGE_LENGTH}-character limit."
        }), 400

    # ── Build the OpenRouter request payload ──────────────────────────────────
    payload = {
        "model": MODEL_ID,
        "messages": [
            # System message sets the assistant's behaviour for every turn.
            {"role": "system", "content": SYSTEM_PROMPT},
            # The user's current spoken message.
            {"role": "user", "content": user_message},
        ],
        # 0.7 balances creativity with coherence for conversational use.
        "temperature": 0.7,
        # 512 tokens ≈ ~380 words — plenty for voice, keeps TTS output short.
        "max_tokens": 512,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter requires HTTP-Referer for free-tier models.
        "HTTP-Referer": "http://localhost:5000",
        # Shown in the OpenRouter usage dashboard under "Applications".
        "X-Title": "Voice Chatbot",
    }

    # ── Call OpenRouter ───────────────────────────────────────────────────────
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout:
        # Server waited 30 seconds with no reply — surface a clear message.
        logger.warning("OpenRouter request timed out after %ds.", REQUEST_TIMEOUT_SECONDS)
        return jsonify({"error": "The AI service took too long to respond. Please try again."}), 504
    except requests.exceptions.ConnectionError as exc:
        # DNS failure, refused connection, etc.
        logger.error("Connection error reaching OpenRouter: %s", exc)
        return jsonify({"error": "Could not reach the AI service. Check your internet connection."}), 502

    # ── Validate the upstream HTTP response ───────────────────────────────────
    if not response.ok:
        # Log a snippet of the body for debugging without overwhelming the log.
        logger.error(
            "OpenRouter returned HTTP %s: %s",
            response.status_code,
            response.text[:400],
        )
        # Translate common HTTP codes into friendly messages.
        if response.status_code == 429:
            return jsonify({"error": "API rate limit reached. Please wait a moment and try again."}), 502
        if response.status_code in (401, 403):
            return jsonify({"error": "API key is invalid or unauthorised."}), 502
        return jsonify({"error": f"AI service returned an error (HTTP {response.status_code})."}), 502

    # ── Extract the reply text safely ─────────────────────────────────────────
    try:
        result = response.json()
        reply_text = result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        # The response shape was not what we expected — log and surface it.
        logger.error("Unexpected OpenRouter response structure: %s | body: %s", exc, response.text[:400])
        return jsonify({"error": "Received an unexpected response from the AI service."}), 502

    # ── Guard against empty replies ────────────────────────────────────────────
    if not reply_text:
        logger.warning("OpenRouter returned an empty reply for message: %r", user_message[:80])
        return jsonify({"error": "The AI returned an empty response. Please try rephrasing."}), 502

    # ── Success ───────────────────────────────────────────────────────────────
    logger.info(
        "Chat OK — user msg: %d chars, reply: %d chars",
        len(user_message),
        len(reply_text),
    )
    return jsonify({"reply": reply_text}), 200


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # debug=True enables hot-reload and the Werkzeug debugger.
    # Never use debug=True in production — use gunicorn instead.
    app.run(debug=True, host="0.0.0.0", port=5000)
