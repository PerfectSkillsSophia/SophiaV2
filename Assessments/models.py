from django.db import models

# Create your models here.
class Recording(models.Model):
    ansId = models.AutoField(primary_key=True)
    submission_id = models.CharField(max_length=255,null=True)
    user_name = models.CharField(max_length=255,null=True)
    assessment_name = models.CharField(max_length=300,null=True)
    assessmenttype = models.CharField(max_length=300,null=True)
    question_id = models.IntegerField(null = True)
    video = models.FileField(upload_to='media',blank=True)
    que = models.CharField(max_length=300,null=True)
    c_ans = models.CharField(max_length=300,null=True)
    recorded_answer = models.CharField(max_length=300,null=True)
    answer_accurecy = models.IntegerField(null=True,default=0)
    confidence=models.IntegerField(null=True,default=0)
    nervousness=models.IntegerField(null=True,default=0)
    neutral=models.IntegerField(null=True,default=0)

    def __str__(self):
        return f"{self.ansId} -{self.user_name}  - {self.assessment_name}"
    

class FinalResult(models.Model):
    submission_id = models.CharField(max_length=255,null=True)
    user_name = models.CharField(max_length=255,null=True)
    assessment_name = models.CharField(max_length=300,null=True)
    assessment_type = models.CharField(max_length=255,null=True,default="None")
    total_accurecy = models.IntegerField(null=True,default=0)
    result_generate = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.submission_id} -{self.assessment_name}  - {self.result_generate}"
    
class Feedback(models.Model):
    user_name = models.CharField(max_length=255,null=True)   
    feedback_type = models.CharField(max_length=100)
    def __str__(self):
        return self.feedback_type

