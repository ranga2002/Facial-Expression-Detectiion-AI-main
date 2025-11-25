import base64
import os
import tempfile
from datetime import datetime
from io import BytesIO

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.db.utils import OperationalError
from django.utils.text import get_valid_filename
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from openai import OpenAI
from PIL import Image

from .forms import ImageUploadForm
from .models import EmotionStreamEntry, WellbeingSurvey
from .utils import (
    DEFAULT_THRESHOLD,
    model_summary,
    preprocess_face,
    predict_emotion,
    get_suggestion,
    summarize_emotion_stream,
    get_model,
)

CHAT_MODEL = "gpt-4o-mini"
MAX_CHAT_TURNS = 10
MAX_ASSISTANT_TOKENS = 220
CRISIS_PHRASES = ("suicide", "self-harm", "hurt myself", "kill myself", "end my life")


def _ensure_session_key(request):
    if not request.session.session_key:
        request.session.create()
    return request.session.session_key


def _crisis_message():
    return (
        "I'm here to help, but I can't provide crisis support. "
        "If you feel unsafe, please contact a trusted person or your local emergency helpline right away."
    )


def _is_crisis(text: str) -> bool:
    lower = (text or "").lower()
    return any(token in lower for token in CRISIS_PHRASES)


def _trim_chat_history(history):
    """
    Keep the latest conversational turns (excluding system messages) to cap memory.
    """
    if not history:
        return []
    system_msgs = [m for m in history if m.get("role") == "system"]
    convo = [m for m in history if m.get("role") != "system"]
    trimmed_convo = convo[-(MAX_CHAT_TURNS * 2) :]
    return (system_msgs[:1] + trimmed_convo) if system_msgs else trimmed_convo


def _build_chat_messages(chat_history, user_name="Friend", last_emotion="Neutral", context_note=None):
    safety_prompt = (
        "You are a supportive wellbeing companion. Be brief (2-4 sentences), warm, and practical. "
        "Do not offer medical advice, diagnosis, or crisis counseling; encourage professional help if risk is implied. "
        f"The user's display name is {user_name} and their last detected emotion is {last_emotion}."
    )
    messages = [{"role": "system", "content": safety_prompt}]
    if context_note:
        messages.append({"role": "system", "content": f"Last check-in summary: {context_note}"})
    for msg in _trim_chat_history(chat_history):
        if msg.get("role") == "system":
            continue
        content = (msg.get("content") or "")[:1200]
        messages.append({"role": msg.get("role"), "content": content})
    return messages


def _serialize_stream_entries(entries):
    serialized = []
    for entry in entries:
        serialized.append(
            {
                "label": entry.label,
                "confidence": float(entry.confidence or 0.0),
                "timestamp": entry.captured_at.isoformat(timespec="seconds"),
                "top": entry.top or [],
            }
        )
    return serialized


def _init_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


client = _init_openai_client()


def _save_uploaded_file(uploaded_file):
    safe_name = get_valid_filename(uploaded_file.name)
    path = default_storage.save(f"uploads/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}", uploaded_file)
    return default_storage.path(path), default_storage.url(path)


def decode_base64_image(data_url):
    if not data_url or ";base64," not in data_url:
        return None

    try:
        _, imgstr = data_url.split(";base64,", 1)
        image_data = base64.b64decode(imgstr)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        filename = f"captured_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        safe_name = get_valid_filename(filename)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        file_path = default_storage.save(f"captures/{safe_name}", ContentFile(buffer.getvalue()))
        return default_storage.path(file_path), default_storage.url(file_path)
    except Exception:
        return None


def home(request):
    return render(request, 'emotion/home.html', {
        'now': datetime.now(),
        'user_name': request.session.get('user_name', 'Friend'),
        'last_emotion': request.session.get('last_emotion')
    })

def about(request):
    return render(request, 'emotion/about.html', {
        'now': datetime.now(),
        'user_name': request.session.get('user_name', 'Friend'),
        'last_emotion': request.session.get('last_emotion')
    })


def model_card(request):
    summary = model_summary()
    return render(request, "emotion/model_card.html", {
        "model_info": summary,
        'now': datetime.now(),
        'user_name': request.session.get('user_name', 'Friend'),
        'last_emotion': request.session.get('last_emotion')
    })


def privacy(request):
    return render(request, "emotion/privacy.html", {
        'now': datetime.now(),
        'user_name': request.session.get('user_name', 'Friend'),
        'last_emotion': request.session.get('last_emotion')
    })


def chat_reply(request):
    if request.method != 'POST':
        return JsonResponse({'reply': "Invalid request method."})

    _ensure_session_key(request)

    if client is None:
        return JsonResponse({'reply': "OpenAI client is not configured. Please set OPENAI_API_KEY."})

    user_message = request.POST.get('message', '').strip()
    if not user_message:
        return JsonResponse({'reply': "Please enter a valid message."})

    if len(user_message) > 800:
        return JsonResponse({'reply': "Please keep messages under 800 characters."})

    name = request.session.get("user_name", "Friend")
    emotion = request.session.get("last_emotion", "Neutral")
    context_note = request.session.get("chat_context")

    chat_history = request.session.get('chat_messages', [])
    chat_history.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now().strftime('%H:%M:%S')
    })

    if _is_crisis(user_message):
        crisis_reply = _crisis_message()
        chat_history.append({
            "role": "assistant",
            "content": crisis_reply,
            "timestamp": datetime.now().strftime('%H:%M:%S')
        })
        request.session['chat_messages'] = _trim_chat_history(chat_history)
        request.session.modified = True
        return JsonResponse({'reply': crisis_reply})

    try:
        ai_reply = generate_openai_response(
            chat_history,
            user_name=name,
            last_emotion=emotion,
            context_note=context_note,
        )
        chat_history.append({
            "role": "assistant",
            "content": ai_reply,
            "timestamp": datetime.now().strftime('%H:%M:%S')
        })
        request.session['chat_messages'] = _trim_chat_history(chat_history)
        request.session.modified = True
        return JsonResponse({'reply': ai_reply})
    except Exception as e:
        return JsonResponse({'reply': f"Sorry, an error occurred: {str(e)}"})

def calculate_score(answers):
    """
    Convert Likert responses (1-5) into a wellbeing score.
    Some questions are negatively phrased, so they are inverted (5 becomes 1).
    """
    reverse_indexes = {1, 5}  # zero-based: question2, question6
    normalized = []
    for index, ans in enumerate(answers):
        try:
            val = int(ans)
        except (TypeError, ValueError):
            val = 3
        val = max(1, min(val, 5))
        if index in reverse_indexes:
            val = 6 - val
        normalized.append(val)

    score = sum(normalized)
    max_score = 5 * len(normalized) if normalized else 0
    percent = int((score / max_score) * 100) if max_score else 0
    return score, percent, max_score


def generate_openai_response(messages, user_name="Friend", last_emotion="Neutral", context_note=None):
    """
    Call OpenAI with safety rails, capped tokens, and short context.
    """
    if client is None:
        return "OpenAI client is not configured. Please set OPENAI_API_KEY."

    last_user_text = ""
    for msg in reversed(messages or []):
        if msg.get("role") == "user":
            last_user_text = msg.get("content") or ""
            break

    if _is_crisis(last_user_text):
        return _crisis_message()

    try:
        prompt = _build_chat_messages(
            chat_history=messages or [],
            user_name=user_name or "Friend",
            last_emotion=last_emotion or "Neutral",
            context_note=context_note,
        )
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=prompt,
            max_tokens=MAX_ASSISTANT_TOKENS,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, I couldn't process your request. Please try again later. ({str(e)})"

def index(request):
    session_key = _ensure_session_key(request)
    try:
        get_model()
    except Exception:
        pass
    try:
        stream_qs = EmotionStreamEntry.objects.filter(session_key=session_key).order_by("-captured_at")[:60]
        stream_entries = _serialize_stream_entries(stream_qs)
    except OperationalError:
        # If migrations haven't run yet, fall back to session-only data.
        stream_entries = request.session.get("emotion_stream", [])
    model_info = model_summary()
    question_count = len([name for name in ImageUploadForm().fields if name.startswith("question")])
    result = {
        "prediction": None,
        "suggestion": None,
        "openai_response": None,
        "score": 0,
        "percent_score": 0,
        "score_max": 5 * question_count,
        "chat_messages": request.session.get('chat_messages', []),
        "remaining_messages": 0,
        "image_preview": None,
        "model_used": model_info.get("name"),
        "top_predictions": [],
        "detector": request.session.get("last_detector"),
        "DEFAULT_THRESHOLD": DEFAULT_THRESHOLD,
        "now": datetime.now(),
        "user_name": request.session.get('user_name', 'Friend'),
        "last_emotion": request.session.get('last_emotion'),
        "stream_summary": summarize_emotion_stream(stream_entries),
        "stream_entries": stream_entries[:12],
        "step_total": 2 + question_count,  # profile/info + questions + comment
        "saved": False,
    }
    image_absolute_path = None
    image_url = None
    processed = None

    if request.method == 'POST':
        if 'clear_chat' in request.POST:
            request.session['chat_messages'] = []
            return redirect('home')

        if 'chat_message' in request.POST:
            user_input = request.POST.get('user_message')
            chat_messages = request.session.get('chat_messages', [])
            if chat_messages and len(chat_messages) < MAX_CHAT_TURNS * 2:
                chat_messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().strftime('%H:%M:%S')
                })
                reply = generate_openai_response(
                    [{"role": m["role"], "content": m["content"]} for m in chat_messages],
                    user_name=request.session.get("user_name", "Friend"),
                    last_emotion=request.session.get("last_emotion", "Neutral"),
                    context_note=request.session.get("chat_context"),
                )
                chat_messages.append({
                    "role": "assistant",
                    "content": reply,
                    "timestamp": datetime.now().strftime('%H:%M:%S')
                })
                request.session['chat_messages'] = _trim_chat_history(chat_messages)
            return redirect('analyze')

        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            name = form.cleaned_data['name']
            gender = form.cleaned_data['gender']
            age = form.cleaned_data['age']
            answers = [
                form.cleaned_data['question1'],
                form.cleaned_data['question2'],
                form.cleaned_data['question3'],
                form.cleaned_data['question4'],
                form.cleaned_data['question5'],
                form.cleaned_data['question6'],
                form.cleaned_data['question7'],
                form.cleaned_data['question8'],
            ]
            comment = form.cleaned_data['comment']

            webcam_image = request.POST.get('captured_image', '').strip()
            uploaded_image = form.cleaned_data.get('image')

            if webcam_image:
                saved = decode_base64_image(webcam_image)
                if saved:
                    image_absolute_path, image_url = saved
                else:
                    form.add_error('image', 'Could not read the captured image. Please try again.')
                    return render(request, 'emotion/result.html', {'form': form, **result})
            elif uploaded_image:
                saved_path = _save_uploaded_file(uploaded_image)
                if saved_path:
                    image_absolute_path, image_url = saved_path
            else:
                form.add_error('image', 'Please upload an image or capture one with the camera.')
                return render(request, 'emotion/result.html', {'form': form, **result})

            if image_absolute_path:
                processed = preprocess_face(image_absolute_path)

            if image_absolute_path and processed:
                prediction_result = predict_emotion(processed["array"], threshold=DEFAULT_THRESHOLD)
                result["prediction"] = prediction_result["label"]
                result["top_predictions"] = prediction_result["top"]
                result["model_used"] = prediction_result.get("model_name")
                result["detector"] = processed.get("detector")
                result["suggestion"] = get_suggestion(result["prediction"])
                result["image_preview"] = processed.get("preview_b64")
                result["score"], result["percent_score"], result["score_max"] = calculate_score(answers)
                stream_entries = _serialize_stream_entries(
                    EmotionStreamEntry.objects.filter(session_key=session_key).order_by("-captured_at")[:60]
                )
                stream_summary = summarize_emotion_stream(stream_entries)
                result["stream_summary"] = stream_summary
                result["stream_entries"] = stream_entries[:12]

                request.session['user_name'] = name
                request.session['last_emotion'] = result["prediction"]
                top_text = ", ".join([f"{lbl} ({conf:.2f})" for lbl, conf in result["top_predictions"]]) or "None"
                summary_prompt = (
                    f"{name}, {gender}, {age} years. "
                    f"Detected emotion: {result['prediction']} "
                    f"(top: {top_text}). "
                    f"Score: {result['score']}/{result['score_max']} ({result['percent_score']}%). "
                    f"Answers: {answers}. Comment: {comment or 'No comment'}. "
                    f"Live stream: dominant {stream_summary.get('dominant')} over {stream_summary.get('total')} frames, "
                    f"coverage {stream_summary.get('coverage')}%, avg conf {stream_summary.get('avg_confidence')}."
                )
                request.session["chat_context"] = summary_prompt
                chat_messages = [{
                    "role": "user",
                    "content": summary_prompt,
                    "timestamp": datetime.now().strftime('%H:%M:%S')
                }]
                first_reply = generate_openai_response(
                    chat_messages,
                    user_name=name,
                    last_emotion=result["prediction"],
                    context_note=summary_prompt,
                )
                chat_messages.append({
                    "role": "assistant",
                    "content": first_reply,
                    "timestamp": datetime.now().strftime('%H:%M:%S')
                })
                result["openai_response"] = first_reply
                result["chat_messages"] = chat_messages
                request.session['chat_messages'] = _trim_chat_history(chat_messages)

                survey = WellbeingSurvey.objects.create(
                    name=name,
                    gender=gender,
                    age=age,
                    answers=answers,
                    comment=comment,
                    prediction=result["prediction"],
                    suggestion=result["suggestion"],
                    model_used=result["model_used"] or "",
                    detector=result["detector"] or "",
                    top_predictions=list(result.get("top_predictions") or []),
                    score=result["score"],
                    percent_score=result["percent_score"],
                    score_max=result["score_max"],
                    stream_summary=stream_summary,
                    client_session_key=session_key,
                    image_path=image_absolute_path or "",
                    image_url=image_url or "",
                    openai_response=first_reply,
                )
                EmotionStreamEntry.objects.filter(session_key=session_key, survey__isnull=True).update(survey=survey)
                result["saved"] = True

            else:
                result["prediction"] = "No face detected"
                result["suggestion"] = "Try another image with a clear, front-facing face."
                request.session['chat_messages'] = []

    else:
        request.session["chat_messages"] = []
        request.session["emotion_stream"] = []
        form = ImageUploadForm()
        result["chat_messages"] = []
        result["stream_summary"] = summarize_emotion_stream(stream_entries)

    result["remaining_messages"] = max(0, MAX_CHAT_TURNS - (len(result["chat_messages"]) // 2))
    return render(request, 'emotion/result.html', {'form': form, **result})


def resources(request):
    wellbeing_tips = [
        ("Box Breathing (4-4-4-4)", "Inhale 4s, hold 4s, exhale 4s, hold 4s. Repeat 4x to reset your nervous system."),
        ("Micro-break", "Stand up, stretch your neck and shoulders, and take 10 deep breaths to reduce screen fatigue."),
        ("Gratitude jot", "Write down three small wins from today. It nudges your brain toward the positive."),
        ("Move for 5", "A 5-minute walk or light mobility routine can lift mood and sharpen focus."),
    ]
    reads = [
        ("WHO: Mental health basics", "Quick primer on stress, anxiety, and when to seek help.", "https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response"),
        ("Mindful breathing guide", "Evidence-backed breathing patterns that calm the body quickly.", "https://www.health.harvard.edu/mind-and-mood/take-a-deep-breath"),
        ("Cognitive reframing", "Simple ways to reframe negative thoughts into balanced ones.", "https://psychologytoday.com/us/blog/in-practice/201301/cognitive-reframing-finding-your-heroic-self"),
    ]
    return render(request, 'emotion/resources.html', {
        'wellbeing_tips': wellbeing_tips,
        'reads': reads,
        'now': datetime.now(),
        'user_name': request.session.get('user_name', 'Friend'),
        'last_emotion': request.session.get('last_emotion')
    })


@csrf_exempt
@require_POST
def stream_frame(request):
    """
    Receive a base64-encoded frame from the browser, run FER, and store lightweight
    predictions in the session so we can summarize during the survey.
    """
    session_key = _ensure_session_key(request)
    frame_data = request.POST.get("frame", "")
    if not frame_data or ";base64," not in frame_data:
        return JsonResponse({"error": "No frame provided."}, status=400)

    tmp_path = None
    try:
        _, imgstr = frame_data.split(";base64,", 1)
        image_bytes = base64.b64decode(imgstr)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        processed = preprocess_face(tmp_path)
        label = "No face detected"
        top = []
        conf = 0.0
        if processed:
            prediction = predict_emotion(processed["array"], threshold=DEFAULT_THRESHOLD)
            label = prediction.get("label", "Unknown")
            top = prediction.get("top", [])
            conf = float(prediction.get("confidence", 0.0))
            request.session["last_detector"] = processed.get("detector")
            request.session.modified = True
        top_for_store = [list(item) for item in top[:3]] if top else []
        entry = {
            "label": label,
            "confidence": conf,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "top": top_for_store,
            "detector": processed.get("detector") if processed else None,
        }
        stream_log = request.session.get("emotion_stream", [])
        stream_log.append(entry)
        # keep only the latest 60 frames to avoid session bloat
        stream_log = stream_log[-60:]
        request.session["emotion_stream"] = stream_log
        request.session.modified = True

        try:
            EmotionStreamEntry.objects.create(
                session_key=session_key,
                label=label,
                confidence=conf,
                top=top_for_store,
            )
            stale = EmotionStreamEntry.objects.filter(session_key=session_key).order_by("-captured_at")[120:]
            if stale:
                EmotionStreamEntry.objects.filter(pk__in=[s.pk for s in stale]).delete()
        except Exception:
            # DB writes should not break the live stream; fail silently but keep session log.
            pass

        summary = summarize_emotion_stream(stream_log)
        return JsonResponse({
            "label": label,
            "confidence": conf,
            "top": top,
            "frames": len(stream_log),
            "summary": summary,
        })
    except Exception as exc:
        return JsonResponse({"error": f"Could not process frame: {str(exc)}"}, status=500)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
