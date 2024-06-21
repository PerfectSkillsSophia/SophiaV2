from django.urls import path
from Assessments.views import * 
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('',Assessments_Dashboard,name='Assessments_Dashboard'),
    path('start_assessments/',Start_Assessments,name='Start_Assessments'),
    path('exam/<str:assessment_code>',Assessments,name='Assessments'),
    path('exam/upload/',Upload_Video,name='Upload_Video'),
    path('feedback_page',feedback_page,name='feedback_page'),
    path('feedback/',feedback,name='feedback'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)