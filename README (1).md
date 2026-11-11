# AI Coach (Streamlit + Ollama)

Choose a topic by **typing, speaking or uploading an image** -> the model explains it in
one paragraph -> it asks 5 summary-style questions -> it grades your answers.
Past sessions are saved in the sidebar history.

## Setup (VS Code)

1. Install Ollama from https://ollama.com and pull the models:
   ```
   ollama pull llama3.2      # text model (explanations, questions, grading)
   ollama pull llava         # vision model (only needed for image input)
   ```
2. Open this folder in VS Code, then open the terminal (Ctrl+`).
3. Create a virtual environment and install packages:
   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS / Linux
   pip install -r requirements.txt
   ```
4. Ctrl+Shift+P -> "Python: Select Interpreter" -> choose `.venv`.
5. Make sure Ollama is running, then start the app:
   ```
   streamlit run app.py
   ```

## Input modes

| Mode | How it works |
|------|--------------|
| Type | Type a topic or a question in the box |
| Speak | Record with the microphone -> `faster-whisper` (local) turns it into text -> the LLM extracts the topic |
| Image | Upload or take a photo -> a vision model (e.g. `llava`) names the topic |

The detected topic is placed in the text box so you can edit it before pressing **Explain it**.

Notes:
- The first time you use speech, Whisper downloads its `base` model (~140 MB). Change
  `WHISPER_SIZE` in `app.py` to `tiny` (faster) or `small` (more accurate).
- Your browser asks for microphone/camera permission; allow it on `localhost`.
- Image input needs a vision-capable model. Pick it in the sidebar ("Vision model").

## Files

- `app.py` - the whole app
- `history.json` - created automatically; stores your past sessions
- `.streamlit/config.toml` - theme colour
