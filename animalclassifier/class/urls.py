from django.urls import path

from . import views

urlpatterns = [
    path('home',views.index,name = 'classifier.index'),
    path('howitworks',views.howitworks,name='classifier.hiw'),
]
