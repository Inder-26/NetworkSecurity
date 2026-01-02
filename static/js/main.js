// ============================================
// Network Security - Main JavaScript
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    initFileUpload();
    initTrainingHandler();
    initAnimations();
    initDynamicWidths();
});

// ============================================
// Training Handler
// ============================================

function initTrainingHandler() {
    const trainBtn = document.getElementById('train-btn');
    if (!trainBtn) return;

    // Create Modal HTML if not exists
    if (!document.getElementById('success-modal')) {
        const modalHtml = `
            <div id="success-modal" class="modal-overlay">
                <div class="modal-content">
                    <button class="close-modal">&times;</button>
                    <div class="modal-icon">
                        <i class="fas fa-check-circle"></i>
                    </div>
                    <h3 class="modal-title">Training Successful!</h3>
                    <p class="modal-message">
                        Model has been trained successfully. You can inspect the detailed metrics and experiments on DagsHub.
                    </p>
                    <div class="modal-actions">
                        <a href="https://dagshub.com/Inder-26/NetworkSecurity/experiments" target="_blank" class="btn btn-primary">
                            <i class="fas fa-external-link-alt"></i>
                            Check Experiments
                        </a>
                    </div>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        
        // Modal Close Logic
        document.querySelector('.close-modal').addEventListener('click', () => {
             document.getElementById('success-modal').classList.remove('show');
             // Optional: reload page when modal is closed to refresh any data
             // window.location.reload(); 
        });
    }

    trainBtn.addEventListener('click', async function() {
        if (!confirm('Are you sure you want to start model training? This may take a few minutes.')) {
            return;
        }

        const originalText = trainBtn.innerHTML;
        trainBtn.disabled = true;
        trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training in Progress...';
        
        try {
            showNotification('Training started. Please wait...', 'info');
            
            const response = await fetch('/api/train', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Show Modal
                const modal = document.getElementById('success-modal');
                modal.classList.add('show');
                
                // Reset button
                 trainBtn.disabled = false;
                 trainBtn.innerHTML = originalText;
                 
            } else {
                throw new Error(data.detail || 'Training failed');
            }
        } catch (error) {
            console.error('Training Error:', error);
            showNotification(`Training failed: ${error.message}`, 'error');
            trainBtn.disabled = false;
            trainBtn.innerHTML = originalText;
        }
    });
}

// ============================================
// File Upload Handling
// ============================================

function initFileUpload() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const removeBtn = document.getElementById('remove-file');
    const submitBtn = document.getElementById('submit-btn');
    const form = document.getElementById('upload-form');

    if (!uploadZone) return;

    // Drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.remove('dragover');
        }, false);
    });

    uploadZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect(files[0]);
        }
    }

    // File input change
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.name.endsWith('.csv')) {
            showNotification('Please upload a CSV file', 'error');
            return;
        }

        fileName.textContent = file.name;
        uploadZone.style.display = 'none';
        fileInfo.style.display = 'flex';
        submitBtn.disabled = false;
    }

    // Remove file
    if (removeBtn) {
        removeBtn.addEventListener('click', function() {
            fileInput.value = '';
            uploadZone.style.display = 'block';
            fileInfo.style.display = 'none';
            submitBtn.disabled = true;
        });
    }

    // Form submit
    if (form) {
        form.addEventListener('submit', function(e) {
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            submitBtn.disabled = true;
        });
    }
}

// ============================================
// Notifications
// ============================================

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.classList.add('show');
    }, 100);
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// ============================================
// Animations
// ============================================

function initAnimations() {
    // Animate elements on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    document.querySelectorAll('.card').forEach(card => {
        observer.observe(card);
    });

    // Animate metric circles
    document.querySelectorAll('.metric-circle').forEach(circle => {
        const progress = circle.style.getPropertyValue('--progress');
        circle.style.setProperty('--progress', '0');
        
        setTimeout(() => {
            circle.style.transition = 'all 1s ease-out';
            circle.style.setProperty('--progress', progress);
        }, 300);
    });
}

function initDynamicWidths() {
    const dynamicElements = document.querySelectorAll('[data-width]');
    
    dynamicElements.forEach(el => {
        const width = el.getAttribute('data-width');
        if (width) {
            // Apply immediately if it's not an animated element, or let CSS transitions handle it
            // We set it as a style property so the browser sees it as valid CSS
            setTimeout(() => {
                el.style.width = `${width}%`;
            }, 100); // Small delay to ensure transitions work if present
        }
    });
}

// ============================================
// Utility Functions
// ============================================

function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}