from django.shortcuts import render, redirect, HttpResponseRedirect
import random
import string
from Administration.models import Assessment , Question
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from Assessments.models import *
from Sophia import settings
from .transcript import upload_and_transcribe_audio ,FindAcc, analyze_video_emotions
from statistics import mean
from django.contrib import messages




# Basic Render Template Views

@staff_member_required(login_url='login')
@login_required(login_url='login')
def Dashboard (request):
		assessment = Assessment.objects
		return render(request,'dashboard.html',{'assessment': assessment})

@staff_member_required(login_url='login')
@login_required(login_url='login')
def Add_Assessments (request):
		return render(request,'add_assessments.html')

@staff_member_required(login_url='login')
@login_required(login_url='login')
def All_Submissions (request):
        url = settings.MEDIA_URL
        all_data = Recording.objects.all().values('user_name', 'assessment_name', 'submission_id','assessmenttype').distinct()
        return render(request,'all_submissions.html',{'all_data':all_data,'url':url})

@staff_member_required(login_url='login')
@login_required(login_url='login')
def Add_Questions (request,ass_id):
		ass_id = ass_id
		assessment = Assessment.objects.filter(assId=ass_id)      
		allque = Assessment.objects.get(assId=ass_id).question_set.all()[:5]
		return render(request,'add_questions.html',{'ques': allque,'assessment': assessment})

@staff_member_required(login_url='login')
@login_required(login_url='login')
def create_Assessments(request):
    if request.method == 'GET':
        assessment_name = request.GET.get('ass_name')

        # Check if an assessment with the same name already exists
        if Assessment.objects.filter(assessment_name=assessment_name).exists():
            messages.error(request, f"Assessment with name '{assessment_name}' already exists.")
        else:
            random_number = random.choice(string.digits)
            random_character = random.choice(string.ascii_letters)
            
            new_ass = Assessment()
            new_ass.assessment_name = assessment_name
            new_ass.assessment_description = request.GET.get('ass_dec')
            new_ass.assessment_code = assessment_name + "_" + random_number + random_character
            new_ass.assessment_type = request.GET.get('ass_type')
            new_ass.save()
            messages.success(request, f"Assessment '{assessment_name}' created successfully.")
        
        return redirect('Add_Assessments')

    return redirect('Add_Assessments')


@staff_member_required(login_url='login')
@login_required(login_url='login')
def Uplaod_Questions(request):
    if request.method == 'GET':
        assessment_code = request.GET.get('assessment_code')
        ass = Assessment.objects.get(assessment_code=assessment_code)
        new_que = Question()
        new_que.question = request.GET.get('question')
        new_que.correctanswer = request.GET.get('answer')
        new_que.assessment = ass = Assessment.objects.get(assessment_code=assessment_code)
        new_que.save()
        return HttpResponseRedirect(reverse("Add_Questions", args=(ass.assId,)))
    return HttpResponseRedirect(reverse("Add_Questions", args=(ass.assId,)))


def Result(request,submission_id):
        url = settings.MEDIA_URL
        recordings = Recording.objects.filter(submission_id=submission_id)
        F_result = FinalResult.objects.filter(submission_id = submission_id)
        return render(request,'result.html',{'recordings':recordings ,'F_result':F_result, 'url':url })


@staff_member_required(login_url='login')
@login_required(login_url='login')
def Generate_Result(request,submission_id):
        total_accurecy = []
        recordings = Recording.objects.filter(submission_id=submission_id)
        video = {recording.ansId: recording.video.path for recording in recordings if recording.video}
        for id, video_file in video.items():
                answer = Recording.objects.get(ansId = id)
                vf = video_file
                confi , nerve , nutral = analyze_video_emotions(vf)
                ra=upload_and_transcribe_audio(vf)
                ca =  Question.objects.get(questionId=answer.question_id).correctanswer
                s1 = ca
                s2 = ra
                Accurecy = FindAcc(s1,s2)
                total_accurecy.append(Accurecy)
                answer.recorded_answer =  ra
                answer.answer_accurecy = Accurecy
                answer.confidence = confi
                answer.nervousness = nerve
                answer.neutral = nutral
                answer.save()
        total_accurecy = mean(total_accurecy)
        new_result = FinalResult.objects.get(submission_id=submission_id)
        new_result.total_accurecy=total_accurecy
        new_result.result_generate=True
        new_result.save()
        return HttpResponseRedirect(reverse("Result", args=(submission_id,)))


