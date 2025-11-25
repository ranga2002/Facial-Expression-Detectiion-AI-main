import base64
import os
import tempfile
from datetime import datetime
from io import BytesIO

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.utils.text import get_valid_filename
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from .forms import ImageUploadForm
from .utils import (
    DEFAULT_THRESHOLD,
    model_summary,
    preprocess_face,
    predict_emotion,
    get_suggestion,
    summarize_emotion_stream,
)

load_dotenv()


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

    if client is None:
        return JsonResponse({'reply': "OpenAI client is not configured. Please set OPENAI_API_KEY."})

    user_message = request.POST.get('message', '').strip()
    if not user_message:
        return JsonResponse({'reply': "Please enter a valid message."})

    # Fetch chat history or start new
    chat_history = request.session.get('chat_messages', [])

    # Add user input
    chat_history.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now().strftime('%H:%M:%S')
    })

    # Add system prompt only once
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
    if not any(msg["role"] == "system" for msg in messages):
        name = request.session.get("user_name", "Friend")
        emotion = request.session.get("last_emotion", "Neutral")
        messages.insert(0, {
            "role": "system",
            "content": f"You are a supportive AI assistant. The user is {name}, feeling {emotion}. Respond warmly and concisely."
        })

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        ai_reply = response.choices[0].message.content.strip()

        chat_history.append({
            "role": "assistant",
            "content": ai_reply,
            "timestamp": datetime.now().strftime('%H:%M:%S')
        })
        request.session['chat_messages'] = chat_history

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


def generate_openai_response(messages):
    if client is None:
        return "OpenAI client is not configured. Please set OPENAI_API_KEY."

    prompt = list(messages)
    prompt.insert(1, {
        "role": "system",
        "content": "You're an AI mental health buddy. Respond with warmth and empathy, briefly (2-4 lines)."
    })
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I couldn't process your request. Please try again later. ({str(e)})"

def index(request):
    stream_entries = request.session.get("emotion_stream", [])
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
        "model_used": None,
        "top_predictions": [],
        "detector": None,
        "DEFAULT_THRESHOLD": DEFAULT_THRESHOLD,
        "now": datetime.now(),
        "user_name": request.session.get('user_name', 'Friend'),
        "last_emotion": request.session.get('last_emotion'),
        "stream_summary": summarize_emotion_stream(stream_entries),
        "stream_entries": list(reversed(stream_entries[-12:])) if stream_entries else [],
        "step_total": 2 + question_count,  # profile/info + questions + comment
    }
    image_absolute_path = None

    if request.method == 'POST':
        if 'clear_chat' in request.POST:
            request.session['chat_messages'] = []
            return redirect('home')

        if 'chat_message' in request.POST:
            user_input = request.POST.get('user_message')
            chat_messages = request.session.get('chat_messages', [])
            if chat_messages and len(chat_messages) < 10:
                chat_messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().strftime('%H:%M:%S')
                })
                reply = generate_openai_response([{"role": m["role"], "content": m["content"]} for m in chat_messages])
                chat_messages.append({
                    "role": "assistant",
                    "content": reply,
                    "timestamp": datetime.now().strftime('%H:%M:%S')
                })
                request.session['chat_messages'] = chat_messages
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
                    image_absolute_path, _ = saved
                else:
                    form.add_error('image', 'Could not read the captured image. Please try again.')
                    return render(request, 'emotion/result.html', {'form': form, **result})
            elif uploaded_image:
                saved_path = _save_uploaded_file(uploaded_image)
                if saved_path:
                    image_absolute_path, _ = saved_path
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
                stream_log = request.session.get("emotion_stream", [])
                stream_summary = summarize_emotion_stream(stream_log)
                result["stream_summary"] = stream_summary
                result["stream_entries"] = list(reversed(stream_log[-12:])) if stream_log else []

                request.session['user_name'] = name
                request.session['last_emotion'] = result["prediction"]

                system_prompt = (
                    f"User info: {name}, {gender}, {age} years old. "
                    f"Top emotion: {result['prediction']}. Score: {result['score']}/{result['score_max']}. "
                    f"Answers: {answers}. Comment: {comment}. "
                    f"Live stream: dominant {stream_summary.get('dominant')} over {stream_summary.get('total')} frames, "
                    f"coverage {stream_summary.get('coverage')}%, avg conf {stream_summary.get('avg_confidence')}."
                )
                chat_messages = [{
                    "role": "system",
                    "content": system_prompt,
                    "timestamp": datetime.now().strftime('%H:%M:%S')
                }]
                first_reply = generate_openai_response([{"role": "system", "content": system_prompt}])
                chat_messages.append({
                    "role": "assistant",
                    "content": first_reply,
                    "timestamp": datetime.now().strftime('%H:%M:%S')
                })
                result["openai_response"] = first_reply
                result["chat_messages"] = chat_messages
                request.session['chat_messages'] = chat_messages
                result["stream_summary"] = summarize_emotion_stream(request.session.get("emotion_stream", []))

            else:
                result["prediction"] = "No face detected"
                result["suggestion"] = "Try another image with a clear, front-facing face."
                request.session['chat_messages'] = []

    else:
        request.session["chat_messages"] = []
        request.session["emotion_stream"] = []
        form = ImageUploadForm()
        result["chat_messages"] = []
        result["stream_summary"] = summarize_emotion_stream([])

    result["remaining_messages"] = max(0, 5 - (len(result["chat_messages"]) // 2))
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

        entry = {
            "label": label,
            "confidence": conf,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "top": top[:3] if top else [],
        }
        stream_log = request.session.get("emotion_stream", [])
        stream_log.append(entry)
        # keep only the latest 60 frames to avoid session bloat
        stream_log = stream_log[-60:]
        request.session["emotion_stream"] = stream_log
        request.session.modified = True

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
