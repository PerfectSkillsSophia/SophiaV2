from django.db import models

class Assessment(models.Model):
    assId = models.AutoField(primary_key=True)
    assessment_name = models.CharField(max_length=255,null=True)
    assessment_code = models.CharField(max_length=300,null=True,default="None")
    assessment_type = models.CharField(max_length=255,null=True,default="None")
    assessment_discription = models.CharField(max_length=1000,null=True)
    def __str__(self):
        return self.assessment_name


class Question(models.Model):
    questionId = models.AutoField(primary_key=True)
    question= models.CharField(max_length=255,null=True)
    correctanswer = models.CharField(max_length=255,null=True)
    assessment = models.ForeignKey(Assessment, on_delete=models.CASCADE)
    def __str__(self):
        return f"ID : {self.questionId}   -  Question_Text : {self.question}"
    
