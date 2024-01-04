from django.urls import path
from Administration.views import * 
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('',Dashboard,name='Dashboard'),
    path('add_assessments/',Add_Assessments,name='Add_Assessments'),
    path('create_assessments/',create_Assessments,name='create_Assessments'),
    path('all_submissions/',All_Submissions,name='All_Submissions'),
    path('add_questions/<int:ass_id>/',Add_Questions,name='Add_Questions'),
    path('result/<str:submission_id>',Result,name='Result'),
    path('result_generation/<str:submission_id>/',Generate_Result,name='Generate_Result'),
    path('uplaod_questions/',Uplaod_Questions,name='Uplaod_Questions'),


]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)