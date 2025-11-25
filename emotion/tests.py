from django.test import Client, TestCase
from django.urls import reverse

from .models import EmotionStreamEntry, WellbeingSurvey
from .views import calculate_score
from .utils import summarize_emotion_stream


class ScoreAndSummaryTests(TestCase):
    def test_calculate_score_respects_reverse_items(self):
        score, percent, max_score = calculate_score(["5"] * 8)
        self.assertEqual(max_score, 40)
        # Two questions are reversed, so perfect agreement becomes slightly lower.
        self.assertLess(score, max_score)
        self.assertGreater(percent, 0)

    def test_summarize_stream(self):
        entries = [
            {"label": "Happy", "confidence": 0.92, "timestamp": "2025-01-01T00:00:00"},
            {"label": "Sad", "confidence": 0.50, "timestamp": "2025-01-01T00:00:01"},
            {"label": "Happy", "confidence": 0.80, "timestamp": "2025-01-01T00:00:02"},
        ]
        summary = summarize_emotion_stream(entries)
        self.assertEqual(summary["dominant"], "Happy")
        self.assertEqual(summary["total"], 3)
        self.assertGreater(summary["coverage"], 0)


class ModelPersistenceTests(TestCase):
    def test_stream_entries_link_to_survey(self):
        survey = WellbeingSurvey.objects.create(
            name="Ada",
            gender="female",
            age=30,
            answers=["3"] * 8,
            prediction="Happy",
            suggestion="Keep smiling",
            model_used="TestModel",
            detector="haar",
            score=24,
            percent_score=60,
            score_max=40,
            stream_summary={"total": 1},
            client_session_key="abc",
        )
        entry = EmotionStreamEntry.objects.create(
            session_key="abc",
            survey=survey,
            label="Happy",
            confidence=0.9,
            top=[["Happy", 0.9]],
        )
        self.assertEqual(survey.stream_entries.count(), 1)
        self.assertEqual(entry.survey_id, survey.id)


class AnalyzeViewTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_get_analyze_page(self):
        resp = self.client.get(reverse("analyze"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Emotion Check-in")
