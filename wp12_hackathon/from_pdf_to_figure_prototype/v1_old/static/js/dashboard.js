// Modern Dashboard JavaScript
class DashboardManager {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.loadRecentAnalyses();
    }

    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.urlInput = document.getElementById('urlInput');
        this.analyzeBtn = document.getElementById('analyzeUrl');
        this.processingSection = document.getElementById('processingSection');
        this.progressFill = document.getElementById('progressFill');
        this.processingStatus = document.getElementById('processingStatus');
    }

    setupEventListeners() {
        // File upload events
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));

        // URL analysis
        this.analyzeBtn.addEventListener('click', this.analyzeUrl.bind(this));
        this.urlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.analyzeUrl();
        });
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type === 'application/pdf') {
            this.processFile(files[0]);
        } else {
            this.showError('Please drop a valid PDF file');
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file && file.type === 'application/pdf') {
            this.processFile(file);
        } else {
            this.showError('Please select a valid PDF file');
        }
    }

    processFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        this.startProcessing(`Analyzing ${file.name}...`);
        this.uploadDocument(formData);
    }

    analyzeUrl() {
        const url = this.urlInput.value.trim();
        if (!url) {
            this.showError('Please enter a valid URL');
            return;
        }

        const formData = new FormData();
        formData.append('url', url);
        
        this.startProcessing('Downloading and analyzing document...');
        this.uploadDocument(formData);
    }

    uploadDocument(formData) {
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                this.simulateProgress(data.redirect);
            } else {
                this.showError(data.error || 'Processing failed');
            }
        })
        .catch(error => {
            this.showError('Upload failed: ' + error.message);
        });
    }

    startProcessing(message) {
        this.processingStatus.textContent = message;
        this.processingSection.style.display = 'block';
        this.uploadArea.style.display = 'none';
        this.resetProgress();
    }

    simulateProgress(redirectUrl) {
        const steps = [
            { id: 'step1', message: 'Analyzing document structure...', progress: 25 },
            { id: 'step2', message: 'Detecting industry type...', progress: 50 },
            { id: 'step3', message: 'Extracting key metrics...', progress: 75 },
            { id: 'step4', message: 'Generating business intelligence...', progress: 100 }
        ];

        let currentStep = 0;
        const stepInterval = setInterval(() => {
            if (currentStep < steps.length) {
                const step = steps[currentStep];
                
                // Update progress
                this.progressFill.style.width = step.progress + '%';
                this.processingStatus.textContent = step.message;
                
                // Update step indicators
                document.getElementById(step.id).classList.add('active');
                
                currentStep++;
            } else {
                clearInterval(stepInterval);
                setTimeout(() => {
                    window.location.href = redirectUrl;
                }, 1000);
            }
        }, 2000);
    }

    resetProgress() {
        this.progressFill.style.width = '0%';
        document.querySelectorAll('.step').forEach(step => {
            step.classList.remove('active');
        });
        document.getElementById('step1').classList.add('active');
    }

    showError(message) {
        // Reset UI
        this.processingSection.style.display = 'none';
        this.uploadArea.style.display = 'block';
        
        // Show error (you could implement a toast notification here)
        alert('Error: ' + message);
    }

    loadRecentAnalyses() {
        // Load recent analyses from localStorage or API
        const recentGrid = document.getElementById('recentGrid');
        const recent = JSON.parse(localStorage.getItem('recentAnalyses') || '[]');
        
        if (recent.length === 0) {
            recentGrid.innerHTML = '<p style="color: rgba(255,255,255,0.7); text-align: center;">No recent analyses</p>';
            return;
        }

        recentGrid.innerHTML = recent.map(analysis => `
            <div class="recent-item" onclick="window.location.href='/results/${analysis.id}'">
                <h3>${analysis.company}</h3>
                <div class="industry-badge" style="font-size: 0.8rem; margin: 0.5rem 0;">
                    ${analysis.industry}
                </div>
                <p style="color: #64748b; font-size: 0.9rem;">
                    ${analysis.metrics} metrics • ${analysis.date}
                </p>
            </div>
        `).join('');
    }

    saveToRecent(analysis) {
        let recent = JSON.parse(localStorage.getItem('recentAnalyses') || '[]');
        recent.unshift(analysis);
        recent = recent.slice(0, 6); // Keep only 6 recent
        localStorage.setItem('recentAnalyses', JSON.stringify(recent));
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DashboardManager();
});

// Add some nice visual effects
document.addEventListener('DOMContentLoaded', () => {
    // Animate metric cards on scroll
    const observeElements = document.querySelectorAll('.metric-card, .insight-card');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animation = 'fadeIn 0.6s ease-out';
            }
        });
    });

    observeElements.forEach(el => observer.observe(el));
});