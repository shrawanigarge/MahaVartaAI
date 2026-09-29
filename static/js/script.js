/**
 * Frontend JavaScript for Marathi News Classification
 * Handles input validation, character counting, UX interactions, and Try Another Headline.
 */

document.addEventListener('DOMContentLoaded', function () {
    const headlineInput = document.getElementById('headlineInput');
    const charCount = document.getElementById('charCount');
    const predictionForm = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictBtn');
    const resultCard = document.getElementById('predictionResultCard');
    const tryAnotherBtn = document.getElementById('tryAnotherBtn');

    // Update character counter
    function updateCharCount() {
        if (headlineInput && charCount) {
            const len = headlineInput.value.length;
            charCount.textContent = `${len} chars`;
        }
    }

    if (headlineInput) {
        headlineInput.addEventListener('input', updateCharCount);
        updateCharCount(); // Initial count
    }

    // "Try Another Headline" button handler
    if (tryAnotherBtn) {
        tryAnotherBtn.addEventListener('click', function () {
            if (headlineInput) {
                headlineInput.value = '';
                updateCharCount();
                headlineInput.focus();
            }
            if (resultCard) {
                // Smoothly fade out and remove result card view
                resultCard.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                resultCard.style.opacity = '0';
                resultCard.style.transform = 'translateY(-10px)';
                setTimeout(() => {
                    resultCard.style.display = 'none';
                }, 300);
            }
            const prepContentArea = document.getElementById('preprocessingContentArea');
            if (prepContentArea) {
                prepContentArea.innerHTML = `
                    <div class="text-center py-5 px-3 bg-light rounded-3 border border-dashed text-muted" id="preprocessingPlaceholder">
                        <i class="bi bi-gear-wide-connected fs-1 text-secondary opacity-50 mb-2 d-block"></i>
                        <p class="mb-0 fw-medium">Enter a Marathi headline above to view its preprocessing.</p>
                    </div>
                `;
            }
            // Scroll to the top of the form
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    // Form submission validation & UI loading state
    if (predictionForm) {
        predictionForm.addEventListener('submit', function (e) {
            const text = headlineInput ? headlineInput.value.trim() : "";
            
            if (!text) {
                e.preventDefault();
                alert("Please enter a Marathi news headline. (कृपया बातमीचे शीर्षक प्रविष्ट करा.)");
                return;
            }

            // Check for Devanagari Unicode characters (\u0900 to \u097F)
            const devanagariRegex = /[\u0900-\u097F]/;
            if (!devanagariRegex.test(text)) {
                e.preventDefault();
                alert("Please enter the headline in Marathi / Devanagari script. (कृपया देवनागरी/मराठी लिपीत बातमी प्रविष्ट करा.)");
                return;
            }

            // Show loading state on button
            if (predictBtn) {
                predictBtn.disabled = true;
                predictBtn.innerHTML = `
                    <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                    Analyzing Marathi Headline...
                `;
            }
        });
    }

    // Auto-scroll to prediction result card if present
    if (resultCard) {
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});
