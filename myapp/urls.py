# myapp/urls.py

from django.urls import path
from .views import index_view, audio_to_text_view

urlpatterns = [
    path('', index_view, name='index'),
    path('audio-to-text/', audio_to_text_view, name='audio_to_text'),
]

