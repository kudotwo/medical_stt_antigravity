// Check for browser support
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (!SpeechRecognition) {
    alert("Your browser does not support the Web Speech API. Please use Google Chrome or Microsoft Edge.");
}

const recognition = new SpeechRecognition();
recognition.continuous = true;
recognition.interimResults = true;

// DOM Elements
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const langSelect = document.getElementById('lang-select');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const finalTextEl = document.getElementById('final-text');
const interimTextEl = document.getElementById('interim-text');
const transcriptBox = document.getElementById('transcript-box');
const loadingOverlay = document.getElementById('loading-overlay');
const reportContent = document.getElementById('report-content');
const btnDownload = document.getElementById('btn-download');

let finalTranscript = '';
let currentSoapData = null;
let isRecording = false;

// Speech Recognition Events
recognition.onstart = function() {
    isRecording = true;
    statusDot.classList.add('recording');
    statusText.textContent = 'Recording...';
    btnStart.disabled = true;
    btnStop.disabled = false;
    
    // Clear previous if any
    finalTranscript = '';
    finalTextEl.innerHTML = '';
    interimTextEl.innerHTML = '';
    
    // Remove placeholder
    const placeholder = transcriptBox.querySelector('.placeholder');
    if (placeholder) placeholder.style.display = 'none';
};

recognition.onresult = function(event) {
    let interimTranscript = '';
    
    for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript + ' ';
        } else {
            interimTranscript += event.results[i][0].transcript;
        }
    }
    
    finalTextEl.innerHTML = finalTranscript;
    interimTextEl.innerHTML = interimTranscript;
    
    // Auto-scroll to bottom
    transcriptBox.parentElement.scrollTop = transcriptBox.parentElement.scrollHeight;
};

recognition.onerror = function(event) {
    console.error("Speech recognition error", event.error);
    statusText.textContent = 'Error: ' + event.error;
    stopRecording();
};

recognition.onend = function() {
    if (isRecording) {
        // If it stopped automatically but we still want to record, restart it
        // (Some browsers stop after a period of silence)
        recognition.start();
    }
};

function stopRecording() {
    isRecording = false;
    recognition.stop();
    statusDot.classList.remove('recording');
    statusText.textContent = 'Analysis mode';
    btnStart.disabled = false;
    btnStop.disabled = true;
}

// Button Listeners
btnStart.addEventListener('click', () => {
    recognition.lang = langSelect.value;
    try {
        recognition.start();
    } catch (e) {
        console.error(e);
    }
});

btnStop.addEventListener('click', () => {
    stopRecording();
    
    // Flush any remaining interim text to final
    if (interimTextEl.innerHTML.trim() !== '') {
        finalTranscript += interimTextEl.innerHTML + ' ';
        finalTextEl.innerHTML = finalTranscript;
        interimTextEl.innerHTML = '';
    }
    
    const textToAnalyze = finalTranscript.trim();
    if (textToAnalyze) {
        analyzeText(textToAnalyze);
    } else {
        alert("No text was recorded.");
        statusText.textContent = 'Ready';
    }
});

btnDownload.addEventListener('click', () => {
    if (!currentSoapData) return;
    downloadCsv(currentSoapData);
});

// API Calls
async function analyzeText(text) {
    loadingOverlay.style.display = 'flex';
    btnDownload.style.display = 'none';
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }
        
        const data = await response.json();
        currentSoapData = data;
        renderSoapReport(data);
        btnDownload.style.display = 'inline-flex';
        statusText.textContent = 'Analysis Complete';
    } catch (error) {
        console.error("Analysis failed", error);
        reportContent.innerHTML = `<p style="color: var(--danger)">Error during analysis: ${error.message}</p>`;
        statusText.textContent = 'Error';
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

async function downloadCsv(soapData) {
    try {
        const response = await fetch('/api/download_csv', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ soap_data: soapData })
        });
        
        if (!response.ok) {
            throw new Error(`Server error generating CSV`);
        }
        
        const csvText = await response.text();
        
        // Create a blob and trigger download
        const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement("a");
        const url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        
        const date = new Date();
        const ts = `${date.getDate()}_${date.getMonth()+1}_${date.getFullYear()}_${date.getHours()}_${date.getMinutes()}`;
        link.setAttribute("download", `medical_report_live_${ts}.csv`);
        
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
    } catch (error) {
        console.error("Download failed", error);
        alert("Failed to generate CSV.");
    }
}

// Render Logic
function renderSoapReport(data) {
    let html = `
        <div class="soap-section">
            <div class="soap-header">Overview</div>
            <div class="soap-content">
                ${makeRow('Summary', data.summary)}
                ${makeRow('Encounter Type', data.encounter_type)}
                ${makeRow('Confidence', data.extraction_confidence)}
            </div>
        </div>
        
        <div class="soap-section">
            <div class="soap-header">Subjective</div>
            <div class="soap-content">
                ${makeRow('Chief Complaint', data.subjective?.chief_complaint)}
                ${makeRow('Symptoms', data.subjective?.symptoms)}
                ${makeRow('Onset', data.subjective?.symptom_onset)}
                ${makeRow('Medical History', data.subjective?.medical_history)}
                ${makeRow('Medications', data.subjective?.current_medications)}
                ${makeRow('Allergies', data.subjective?.allergies)}
            </div>
        </div>
        
        <div class="soap-section">
            <div class="soap-header">Objective</div>
            <div class="soap-content">
                ${makeRow('Vitals', data.objective?.vital_signs)}
                ${makeRow('Physical Exam', data.objective?.physical_exam_findings)}
            </div>
        </div>
        
        <div class="soap-section">
            <div class="soap-header">Assessment & Plan</div>
            <div class="soap-content">
                ${makeRow('Diagnosis', data.assessment?.diagnosis)}
                ${makeRow('Differentials', data.assessment?.differential_diagnosis)}
                ${makeRow('Prescriptions', data.plan?.prescribed_medications)}
                ${makeRow('Treatment Plan', data.plan?.treatment_plan)}
                ${makeRow('Investigations', data.plan?.investigations_ordered)}
                ${makeRow('Referrals', data.plan?.referrals)}
                ${makeRow('Follow up', data.plan?.follow_up)}
            </div>
        </div>
    `;
    
    reportContent.innerHTML = html;
}

function makeRow(label, value) {
    if (value === null || value === undefined || value === '') return '';
    if (Array.isArray(value)) {
        if (value.length === 0) return '';
        value = value.join(', ');
    }
    return `
        <div class="soap-row">
            <div class="soap-label">${label}</div>
            <div class="soap-value">${value}</div>
        </div>
    `;
}
