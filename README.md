# Facial Expression Detection AI

Django web app that blends facial expression recognition with a short wellbeing pulse and an optional OpenAI chat companion. Capture from webcam or upload, preprocess faces locally, score emotions with pre-trained CNNs, and stream a concise reflection back to the user.

---

## What it does
- Capture or upload a face image; preprocess with CLAHE and either MTCNN (preferred) or Haar cascades.
- Stream 1 frame/sec from the browser while the user answers questions; only predictions are kept in session (capped to ~60 frames).
- Run the first available model in `fer_model` (AffectNet -> CK+ JSON/weights -> CK+-based -> legacy FER) with automatic label alignment.
- Score an 8-question Likert wellbeing survey (Q2 and Q6 reverse-scored) to produce raw and percent scores.
- Seed a short GPT-4o-mini response and keep the chat history in-session when `OPENAI_API_KEY` is set.
- Responsive Bootstrap + AOS UI with camera modal, chat modal, and result cards for predictions, scores, and live stream summaries.

---

## Requirements
- Python 3.10+
- pip and virtualenv (recommended)
- Optional: GPU/CUDA for faster TensorFlow inference; OpenAI API key for the chat companion

---

## Quickstart
1) Clone and enter the project
```bash
git clone https://github.com/ranga2002/Facial-Expression-Detectiion-AI.git
cd Facial-Expression-Detectiion-AI
```

2) Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # macOS/Linux
```

3) Install dependencies
```bash
pip install -r requirements.txt
```

4) Configure environment variables (create `.env` in the repo root)
```
OPENAI_API_KEY=your-openai-key          # optional; enables chat and auto-replies
DJANGO_SECRET_KEY=change-me             # required; set a strong value in production
DJANGO_DEBUG=true                       # set to false in production
DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost
```

5) Run migrations and start the dev server
```bash
python manage.py migrate
python manage.py runserver
```
Visit http://127.0.0.1:8000/ to start analyzing emotions.

---

## Using the app
- Home -> Analyze: allow camera access if you want live 1 fps capture during the survey; otherwise upload a clear, front-facing photo.
- Stepper form: provide name, gender, age, and answer the 8 Likert questions (Q2 and Q6 are reverse-scored).
- Submit: the server crops the face, runs the best-available model, shows top predictions, a preview crop, wellbeing score, and live stream stats (dominant label, coverage, avg/std confidence).
- Chat: if `OPENAI_API_KEY` is set, a first reply is generated from your inputs; continue the conversation in the floating chat modal.

---

## Environment variables
- `OPENAI_API_KEY` (optional): enables GPT-4o-mini chat replies.
- `DJANGO_SECRET_KEY`: Django secret key (required in production).
- `DJANGO_DEBUG`: `true`/`false` toggle for debug mode.
- `DJANGO_ALLOWED_HOSTS`: comma-separated hostnames when `DEBUG=false`.

---

## Model artifacts
Store models in `fer_model/` (present in this repo). The loader picks the first available in this order:
1) `AffectNet_trained_keras.h5` (8 labels, preferred)
2) `Pure CK+48.json` + `Pure CK+48_weights.h5`
3) `CK+-based.h5`
4) `emotion_model.h5` (legacy FER2013, 6 labels)

If none load, predictions fall back to `"Model unavailable"`.

---

## API surface
- `POST /api/frame/` - accepts `frame` as a base64 data URL for live stream scoring; returns the latest label, confidence, top logits, and summary stats.
- `POST /chat-reply/` - accepts `message` (form-encoded) and returns a `reply` string; requires `OPENAI_API_KEY`.

---

## Project layout
```
emotion/           # Django app: forms, views, utils, templates
fer_django_app/    # Django project settings/urls
fer_model/         # Pre-trained model files (.h5, .json, weights)
media/             # Uploaded/captured images (local dev storage)
manage.py
requirements.txt
Facial_Emotion_Recognition_Final.ipynb  # training/experiments
```

---

## Troubleshooting
- Seeing "Model unavailable": confirm at least one `.h5` (or CK+ JSON/weights) file exists in `fer_model/` and matches your TensorFlow environment.
- MTCNN errors: `mtcnn` is optional; the app will fall back to Haar cascades automatically.
- Chat disabled: set `OPENAI_API_KEY` and restart; the UI will respond with a friendly warning if the key is missing.
- Webcam blocked: allow camera permissions or use the upload path; live stream is optional.

---

## Security and deployment
- Keep `.env` out of version control; rotate any keys that were ever committed.
- Set `DJANGO_DEBUG=false`, configure `DJANGO_ALLOWED_HOSTS`, and use HTTPS in production.
- Run `python manage.py collectstatic` behind a CDN or static host; ensure media storage is secured if you persist uploads.

---

## Dataset and training
- Models are trained/fine-tuned on [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) and CK+ variants. See `Facial_Emotion_Recognition_Final.ipynb` for experiments and preprocessing steps.

---

For feedback or collaboration: `chakilamsriranga@gmail.com`

_Built to make mental health check-ins quicker and kinder._
