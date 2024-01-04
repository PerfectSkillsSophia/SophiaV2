from django.shortcuts import render,redirect
from Accounts.forms import UserRegistrationForm
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout


def Home (request):
	if request.user.is_authenticated:
		if request.user.is_staff:
			return redirect('Dashboard')
		return redirect('Assessments_Dashboard')
	else:
		return render(request,'index.html')
	



		


def signIn (request):
	if request.user.is_authenticated:
		if request.user.is_staff:
			return redirect('Dashboard')
		return redirect('Assessments_Dashboard')
	else:
		if request.method == 'POST':
			username = request.POST.get('username')
			password =request.POST.get('password')
			user = authenticate(request, username=username, password=password)
			if user is not None:
				login(request, user)
				if request.user.is_staff:
					return redirect('Dashboard')
				return redirect('Home')
			else:
				messages.warning(request, 'Incorrect Credentials ')
		return render(request, 'signin.html')








def signUp(request):
	if request.user.is_authenticated:
		if request.user.is_staff:
			return redirect('Dashboard')
		return redirect('Assessments_Dashboard')
	else:
		form = UserRegistrationForm()
		if request.method == 'POST':
			form = UserRegistrationForm(request.POST)
			if form.is_valid():
				form.save()
				return redirect('login')
		return render(request, 'signup.html')











def logoutUser(request):
	logout(request)
	return render(request,'index.html')