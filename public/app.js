const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const browseLink = document.querySelector('.browse-link');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const removeBtn = document.getElementById('remove-btn');
const predictBtn = document.getElementById('predict-btn');
const resultsSection = document.getElementById('results-section');
const loader = document.querySelector('.loader');
const btnText = document.querySelector('.btn-text');

let currentFile = null;

// Drag & Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// Click to Upload
dropZone.addEventListener('click', (e) => {
    if (e.target !== removeBtn) fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewContainer.style.display = 'block';
        document.querySelector('.upload-content').style.display = 'none';
        predictBtn.disabled = false;
        resultsSection.style.display = 'none'; // Clear previous results
    };
    reader.readAsDataURL(file);
}

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    currentFile = null;
    fileInput.value = '';
    previewContainer.style.display = 'none';
    document.querySelector('.upload-content').style.display = 'block';
    predictBtn.disabled = true;
    resultsSection.style.display = 'none';
});

// Predict
predictBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // UI Loading State
    predictBtn.disabled = true;
    btnText.style.display = 'none';
    loader.style.display = 'inline-block';
    resultsSection.style.display = 'none';

    const formData = new FormData();
    formData.append('file', currentFile);
    
    // Get selected model
    const modelType = document.querySelector('input[name="model_type"]:checked').value;
    formData.append('model_type', modelType);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }

        const data = await response.json();
        renderResults(data);

    } catch (error) {
        alert(error.message);
    } finally {
        predictBtn.disabled = false;
        btnText.style.display = 'inline-block';
        loader.style.display = 'none';
    }
});

function renderResults(data) {
    resultsSection.innerHTML = `
        <div class="result-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h2>${data.class}</h2>
                <div class="badge" style="background: var(--success); margin: 0;">${(data.confidence * 100).toFixed(1)}% Confidence</div>
            </div>
            <p style="color: var(--text-muted); margin-top: 0.5rem; font-size: 0.9rem;">
                Model: ${data.model_used === 'baseline' ? 'Baseline CNN' : 'ResNet18 Fine-Tuned'} | Time: ${data.inference_time_ms}ms
            </p>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: 0%"></div>
            </div>
            
            <div style="margin-top: 1.5rem; display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.8rem; color: var(--text-muted);">
                 ${Object.entries(data.probabilities)
                    .sort(([,a], [,b]) => b - a)
                    .slice(0, 4)
                    .map(([key, val]) => `
                        <div>${key}</div>
                        <div style="text-align: right;">${(val * 100).toFixed(1)}%</div>
                    `).join('')}
            </div>
        </div>
    `;
    
    resultsSection.style.display = 'block';
    
    // Animate bar
    setTimeout(() => {
        document.querySelector('.confidence-fill').style.width = `${data.confidence * 100}%`;
    }, 100);
}
