from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = "mainpage"
urlpatterns = [
    path('', views.index, name='index')
]