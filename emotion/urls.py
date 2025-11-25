from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),         # Homepage
    path('analyze/', views.index, name='analyze'),  # Emotion analysis page
    path('about/', views.about, name='about'),  # About page
    path('model/', views.model_card, name='model_card'),  # Model + data page
    path('privacy/', views.privacy, name='privacy'),  # Privacy page
    path('chat-reply/', views.chat_reply, name='chat_reply'),  # Chat reply endpoint
    path('resources/', views.resources, name='resources'),  # Wellness resources
    path('api/frame/', views.stream_frame, name='frame_api'),  # Live frame intake
]
