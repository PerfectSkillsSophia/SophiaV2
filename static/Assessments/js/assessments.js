
/* var questionContainers = document.querySelectorAll('.question-container');
    var nextQuestionBtn = document.getElementById('nextQuestionBtn');
    var currentQuestionIndex = 0;

    function showNextQuestion() {
        // Hide the current question
        questionContainers[currentQuestionIndex].style.display = 'none';

        // Increment the question index
        currentQuestionIndex++;

        // If all questions are shown, reset to the first question
        if (currentQuestionIndex === questionContainers.length) {
            currentQuestionIndex = 0;
            nextQuestionBtn.textContent = 'Submit'; // Change button text to "Submit"
        }

        // Show the next question
        questionContainers[currentQuestionIndex].style.display = 'block';
    } */


  
    
document.addEventListener('DOMContentLoaded', () => {
    const yourItems = document.querySelectorAll('.model-item');
    const question_ids = [];

    yourItems.forEach(function(currentElement) {
      const itemName = currentElement.textContent.trim();  // Extracting the text content
      question_ids.push(itemName);
    });

    var questionContainers = document.querySelectorAll('.question-container');
    var popupWindow = document.getElementById('popupWindow');
    var currentQuestionIndex = 0;
    const videoElement = document.getElementById('video');
    const startButton = document.getElementById('startButton');
    const nextButton = document.getElementById('nextQuestionBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const ass_name = document.getElementById("assname").innerHTML;
    let mediaRecorder;
    let recordedChunks = [];
    let blobList = [];

    // Check if the browser supports getUserMedia
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        // Request permission to use both camera and audio
        navigator.mediaDevices.getUserMedia({ video: true, audio: true })
            .then((stream) => {
                // Set the video source to the stream
                videoElement.srcObject = stream;

                // Create a MediaRecorder for video and audio
                mediaRecorder = new MediaRecorder(stream);

                // Event listener for when recording starts
                mediaRecorder.onstart = handleRecordingStart;

                // Event listener for when video data is available
                mediaRecorder.ondataavailable = handleDataAvailable;

                // Event listener for when recording stops
                mediaRecorder.onstop = handleRecordingStop;

                // Start recording when the "Start" button is clicked
                startButton.addEventListener('click', startRecordingbtn);

                // Stop current recording, create a blob, and start a new recording when "Next" is clicked
                nextButton.addEventListener('click', nextRecording);
            })
            .catch((error) => {
                console.error('Error accessing camera and microphone:', error);
            });
    } else {
        console.error('getUserMedia is not supported in this browser');
    }

    function startRecordingbtn() {
        // Start recording
        if (startButton.textContent == 'Stop') {
            mediaRecorder.stop();
            nextButton.disabled = true;
        }
        mediaRecorder.start();
        startButton.disabled = true;
        nextButton.disabled = false;

        // Disable the "Start" button during recording
    }

    function handleRecordingStart() {
        if (recordingStatus) {
            recordingStatus.textContent = 'Recording...';
        }
        // Change the text of the "Start" button to "Stop" during recording
        startButton.textContent = 'Stop';
    }

    function handleDataAvailable(event) {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    }

    function handleRecordingStop() {
        if (recordingStatus) {
            recordingStatus.textContent = 'Not Recording';
        }
        // Change the text of the "Start" button back to "Start" after recording stops
        // Enable the "Next" button after recording stops


        // Create a blob from the recorded chunks
        const blob = new Blob(recordedChunks, { type: 'video/webm' });

        // Add the blob to the blob list
        blobList.push(blob);

        // Reset the recorded chunks for the next recording
        recordedChunks = [];
    }

    function nextRecording() {

        if (nextButton.textContent == 'Submit') {
            submit()
        }
        else {
            mediaRecorder.stop();
            mediaRecorder.start();
            // Hide the current question
            questionContainers[currentQuestionIndex].style.display = 'none';

            // Increment the question index
            currentQuestionIndex++;

            // If all questions are shown, disable the button and change its text to "Submit"
            if (currentQuestionIndex === questionContainers.length - 1) {
                nextQuestionBtn.disabled = true;
                nextQuestionBtn.textContent = 'Submit'; // Change button text to "Submit"
                // Disable the button on the last question
                startButton.disabled = false;
            }
            // Show the next question
            questionContainers[currentQuestionIndex].style.display = 'block';
        }

    }



    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
          const cookies = document.cookie.split(';');
          for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
              cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
              break;
            }
          }
        }
        return cookieValue;
      }



    function submit() {
        // Log the list of blobs to the console
        console.log('Blob List:', blobList);
    
        const container = new FormData();
        container.append('ass_name', ass_name);
        container.append('question_ids', question_ids);
        for (let i = 0; i < blobList.length; i++) {
            // Append each blob with a unique key (e.g., 'blob0', 'blob1', etc.)
            container.append(`blob${i}`, blobList[i]);
        }
    
        // Log each entry in the FormData object
        for (const entry of container.entries()) {
            console.log(entry);
        }

        $.ajax({
            type: 'POST',
            url: 'upload/',
            headers: { "X-CSRFToken": getCookie("csrftoken") },
            mode: "same-origin",
            beforeSend: function () {
                alert("Please press OK to conformation and sit back and wait till next instruction...!");
                nextQuestionBtn.disabled = true;
                startButton.disabled = true;

            },
            success: function () {

                popupWindow.style.display = 'block'

            },
            error: function () {
              alert("Error occurred. Please try again.");

            },
            data: container,
            processData: false,
            contentType: false
          });







    }
    

});
