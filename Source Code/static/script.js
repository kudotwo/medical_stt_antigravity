// ── Auth: redirect to login on 401 ──────────────────────────────────────────
const _originalFetch = window.fetch.bind(window);
window.fetch = async function(...args) {
    const response = await _originalFetch(...args);
    if (response.status === 401) {
        window.location.href = '/login';
        return response;
    }
    return response;
};

// ── Logout button ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const btnLogout = document.getElementById('btn-logout');
    if (btnLogout) {
        btnLogout.addEventListener('click', async () => {
            await fetch('/api/logout', { method: 'POST', credentials: 'include' });
            window.location.href = '/login';
        });
    }
});

// Check for browser support
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

let recognition = null;
if (!SpeechRecognition) {
    console.warn("Web Speech API not supported. Recording disabled.");
} else {
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
}

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
const btnEdit = document.getElementById('btn-edit');
const reportActions = document.getElementById('report-actions');

let finalTranscript = '';
let currentSoapData = null;
let isRecording = false;
let isEditMode = false;

// Speech Recognition Events
if (recognition) {
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
            recognition.start();
        }
    };
}

function stopRecording() {
    isRecording = false;
    if (recognition) recognition.stop();
    statusDot.classList.remove('recording');
    statusText.textContent = 'Analysis mode';
    btnStart.disabled = false;
    btnStop.disabled = true;
}

// Button Listeners
btnStart.addEventListener('click', () => {
    if (!recognition) {
        alert('Speech recognition is not supported in this browser. Please use Chrome or Edge.');
        return;
    }
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
    // If in edit mode, sync edits to currentSoapData first
    if (isEditMode) {
        syncEditsToData();
    }
    downloadCsv(currentSoapData);
});

// Edit / Done Editing toggle
btnEdit.addEventListener('click', () => {
    if (!isEditMode) {
        enterEditMode();
    } else {
        exitEditMode();
    }
});

// ── Edit Mode Logic ──────────────────────────────────────────────────────────

function enterEditMode() {
    isEditMode = true;
    btnEdit.innerHTML = '<i class="fa-solid fa-check"></i> Done Editing';
    btnEdit.classList.remove('secondary');
    btnEdit.classList.add('edit-active');

    // Make all soap-value elements contenteditable
    reportContent.querySelectorAll('.soap-value').forEach(el => {
        el.contentEditable = 'true';
        el.classList.add('editable');
        el.setAttribute('spellcheck', 'false');
    });

    // Show the edit hint banner
    let hint = reportContent.querySelector('.edit-hint');
    if (!hint) {
        hint = document.createElement('div');
        hint.className = 'edit-hint';
        hint.innerHTML = '<i class="fa-solid fa-circle-info"></i> Click any field to edit. Changes will be included in the CSV export.';
        reportContent.prepend(hint);
    }
    hint.style.display = 'flex';
}

function exitEditMode() {
    isEditMode = false;
    btnEdit.innerHTML = '<i class="fa-solid fa-pen-to-square"></i> Edit Report';
    btnEdit.classList.add('secondary');
    btnEdit.classList.remove('edit-active');

    // Remove contenteditable
    reportContent.querySelectorAll('.soap-value').forEach(el => {
        el.contentEditable = 'false';
        el.classList.remove('editable');
    });

    // Sync edits back into the data object
    syncEditsToData();

    // Hide the hint
    const hint = reportContent.querySelector('.edit-hint');
    if (hint) hint.style.display = 'none';
}

/**
 * Reads the current text content of each soap-value element in the DOM and
 * writes it back to currentSoapData using the data-field-path attribute.
 */
function syncEditsToData() {
    if (!currentSoapData) return;

    reportContent.querySelectorAll('.soap-value[data-field-path]').forEach(el => {
        const path = el.getAttribute('data-field-path');
        const newValue = el.innerText.trim();
        setNestedValue(currentSoapData, path, newValue);
    });
}

/**
 * Sets a value in a nested object using a dot-separated path string.
 * e.g. setNestedValue(obj, "subjective.chief_complaint", "headache")
 */
function setNestedValue(obj, path, value) {
    const keys = path.split('.');
    let current = obj;
    for (let i = 0; i < keys.length - 1; i++) {
        if (current[keys[i]] === undefined || current[keys[i]] === null) return;
        current = current[keys[i]];
    }
    const lastKey = keys[keys.length - 1];
    // Preserve array type if the original was an array
    if (Array.isArray(current[lastKey])) {
        // Split by comma if the user typed a comma-separated list
        current[lastKey] = value.split(',').map(s => s.trim()).filter(s => s !== '');
    } else {
        current[lastKey] = value;
    }
}

// ── API Calls ────────────────────────────────────────────────────────────────

async function analyzeText(text) {
    loadingOverlay.style.display = 'flex';
    reportActions.style.display = 'none';

    // Exit edit mode if we're re-analyzing
    if (isEditMode) {
        isEditMode = false;
        btnEdit.innerHTML = '<i class="fa-solid fa-pen-to-square"></i> Edit Report';
        btnEdit.classList.add('secondary');
        btnEdit.classList.remove('edit-active');
    }
    
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
        reportActions.style.display = 'flex';
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

// ── Render Logic ─────────────────────────────────────────────────────────────

function renderSoapReport(data) {
    let html = `
        <div class="soap-section">
            <div class="soap-header">Overview</div>
            <div class="soap-content">
                ${makeRow('Summary', data.summary, 'summary')}
                ${makeRow('Encounter Type', data.encounter_type, 'encounter_type')}
                ${makeRow('Confidence', data.extraction_confidence, 'extraction_confidence')}
            </div>
        </div>
        
        <div class="soap-section">
            <div class="soap-header">Subjective</div>
            <div class="soap-content">
                ${makeRow('Chief Complaint', data.subjective?.chief_complaint, 'subjective.chief_complaint')}
                ${makeRow('Symptoms', data.subjective?.symptoms, 'subjective.symptoms')}
                ${makeRow('Onset', data.subjective?.symptom_onset, 'subjective.symptom_onset')}
                ${makeRow('Medical History', data.subjective?.medical_history, 'subjective.medical_history')}
                ${makeRow('Medications', data.subjective?.current_medications, 'subjective.current_medications')}
                ${makeRow('Allergies', data.subjective?.allergies, 'subjective.allergies')}
            </div>
        </div>
        
        <div class="soap-section">
            <div class="soap-header">Objective</div>
            <div class="soap-content">
                ${makeRow('Vitals', data.objective?.vital_signs, 'objective.vital_signs')}
                ${makeRow('Physical Exam', data.objective?.physical_exam_findings, 'objective.physical_exam_findings')}
            </div>
        </div>
        
        <div class="soap-section">
            <div class="soap-header">Assessment & Plan</div>
            <div class="soap-content">
                ${makeRow('Diagnosis', data.assessment?.diagnosis, 'assessment.diagnosis')}
                ${makeRow('Differentials', data.assessment?.differential_diagnosis, 'assessment.differential_diagnosis')}
                ${makeRow('Prescriptions', data.plan?.prescribed_medications, 'plan.prescribed_medications')}
                ${makeRow('Treatment Plan', data.plan?.treatment_plan, 'plan.treatment_plan')}
                ${makeRow('Investigations', data.plan?.investigations_ordered, 'plan.investigations_ordered')}
                ${makeRow('Referrals', data.plan?.referrals, 'plan.referrals')}
                ${makeRow('Follow up', data.plan?.follow_up, 'plan.follow_up')}
            </div>
        </div>
    `;
    
    reportContent.innerHTML = html;
}

/**
 * Renders a single label/value row.
 * @param {string} label   - Display label
 * @param {*}      value   - The data value (string, array, or null)
 * @param {string} path    - Dot-separated path used to write back edits (e.g. "subjective.chief_complaint")
 */
function makeRow(label, value, path) {
    if (value === null || value === undefined || value === '') return '';
    if (Array.isArray(value)) {
        if (value.length === 0) return '';
        value = value.join(', ');
    }
    return `
        <div class="soap-row">
            <div class="soap-label">${label}</div>
            <div class="soap-value" data-field-path="${path}">${value}</div>
        </div>
    `;
}
