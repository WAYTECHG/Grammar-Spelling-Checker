document.addEventListener('DOMContentLoaded', function() {
    const textInput = document.getElementById('text-input');
    const correctedTextDiv = document.getElementById('corrected-text');
    const issuesListDiv = document.getElementById('issues-list');
    const quickCheckBtn = document.getElementById('quick-check');
    const deepCheckBtn = document.getElementById('deep-check');
    const exportBtn = document.getElementById('export');
    const loadingSpinner = document.getElementById('loading-spinner');
    
    let originalCorrections = [];
    let originalText = '';
    
    quickCheckBtn.addEventListener('click', function() {
        if (!textInput.value.trim()) {
            alert('Please enter some text to check.');
            return;
        }
        
        originalText = textInput.value; // Store original text
        performCheck('quick');
    });
    
    deepCheckBtn.addEventListener('click', function() {
        let textToCheck = '';

        // Prefer using the corrected text (from previous layer), if available
        if (correctedTextDiv.textContent.trim()) {
            textToCheck = correctedTextDiv.textContent.trim(); // Remove all span markup
        } else {
            textToCheck = textInput.value.trim();
        }
    
        if (!textToCheck) {
            alert('Please enter some text to check.');
            return;
        }
    
        performCheck('deep', textToCheck);
    
    });
    
    exportBtn.addEventListener('click', function() {
        if (!correctedTextDiv.textContent.trim()) {
            alert('No corrected text to export.');
            return;
        }
        
        const blob = new Blob([correctedTextDiv.textContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'corrected-text.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });
    
    function performCheck(mode, overrideText = null) {
        showLoading();
        
        const endpoint = mode === 'quick' ? '/check' : '/deepcheck';
        const formData = new FormData();
        const textToSend = overrideText || textInput.value;
        formData.append('text', textToSend);
        
        fetch(endpoint, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            displayResults(data, mode);
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            alert('An error occurred while processing your text. Please try again.');
        });
    }
    
    function displayResults(data, mode) {
        // Store the original text for potential deep check later
        originalText = data.original_text;
        
        // Display the corrected text
        displayCorrections(data.final_corrected_text, data.corrections, mode);

        // Inside displayResults()
        if (mode === 'quick') {
            textInput.value = data.final_corrected_text; // Optional sync
        }

    }
    
    function mergeCorrections(corrections) {
        const sorted = [...corrections].sort((a, b) => (a.start_index || 0) - (b.start_index || 0));
        const merged = [];
    
        let lastEnd = -1;
    
        sorted.forEach(corr => {
            const start = corr.start_index || 0;
            const length = corr.original?.length || 0;
            const end = start + length;
    
            if (start >= lastEnd) {
                merged.push(corr);
                lastEnd = end;
            } else {
                // Overlap — replace if better (longer original)
                const prev = merged[merged.length - 1];
                const prevLength = prev.original?.length || 0;
    
                if (length > prevLength) {
                    merged[merged.length - 1] = corr;
                    lastEnd = end;
                }
            }
        });
    
        return merged;
    }
    
    function recalculateIndices(originalText, corrections) {
        let usedPositions = new Set();
    
        return corrections.map(correction => {
            const target = correction.original?.trim();
            if (!target) return correction;
    
            // Search all occurrences of target
            let idx = -1;
            let startSearch = 0;
    
            while (startSearch < originalText.length) {
                idx = originalText.indexOf(target, startSearch);
    
                if (idx === -1) break;
    
                // If this occurrence not used yet, pick it
                if (![...Array(target.length).keys()].every(offset => usedPositions.has(idx + offset))) {
                    // Mark positions as used
                    for (let i = 0; i < target.length; i++) {
                        usedPositions.add(idx + i);
                    }
    
                    // Update correction
                    return {
                        ...correction,
                        start_index: idx
                    };
                }
    
                // Else, search further
                startSearch = idx + 1;
            }
    
            // Fallback: return as-is if not found
            return correction;
        });
    }
    

    function displayCorrections(correctedText, corrections, mode) {
        console.log("DEBUG baseText:", originalText);
    
        if (!corrections || corrections.length === 0) {
            correctedTextDiv.textContent = correctedText;
            issuesListDiv.innerHTML = '<p>No issues found.</p>';
            return;
        }
    
        originalCorrections = corrections;
        const recalculated = recalculateIndices(originalText, corrections);
        const merged = mergeCorrections(recalculated);
        console.log("DEBUG recalculated corrections:", recalculated);
    
        if (mode === 'deep') {
            correctedTextDiv.textContent = correctedText;
    
            let issuesHtml = '<ul class="issues-list">';
            merged.forEach((correction, index) => {
                issuesHtml += `
                    <li class="issue-item ${correction.type}">
                        <div class="issue-header">
                            <span class="issue-type">${correction.type.toUpperCase()}</span>
                            <span class="issue-source">${correction.source || 'Layer 2'}</span>
                            <span class="issue-confidence">Confidence: ${correction.confidence?.toFixed(2)}</span>
                        </div>
                        <div class="issue-content">
                            <div class="original-text">Original: "${correction.original}"</div>
                            <div class="corrected-text">Corrected: "${correction.corrected || correction.suggestion}"</div>
                        </div>
                    </li>
                `;
            });
            issuesHtml += '</ul>';
            issuesListDiv.innerHTML = issuesHtml;
            return;
        }
    
        // --- QUICK CHECK SAFE ---
        const sortedCorrections = [...merged].sort((a, b) => (a.start_index || 0) - (b.start_index || 0));
        console.log("DEBUG sortedCorrections:", sortedCorrections);
    
        let segments = [];
        let lastIndex = 0;
    
        sortedCorrections.forEach((correction, index) => {
            const start = correction.start_index || 0;
            const length = correction.original?.length || 0;
            const end = start + length;
            const corrected = correction.corrected || correction.suggestion || correction.original;
            const type = correction.type || 'grammar';
    
            // Text before this correction
            if (lastIndex < start) {
                const plainText = originalText.slice(lastIndex, start);
                segments.push(plainText);
            }
    
            // Correction span
            const span = `<span class="error ${type}" id="correction-${index}" data-correction-index="${index}" title="${type}">${corrected}</span>`;
            segments.push(span);
    
            lastIndex = start + length; // move index forward correctly
        });
    
        // Remaining text
        if (lastIndex < originalText.length) {
            segments.push(originalText.slice(lastIndex));
        }
    
        const resultHtml = segments.join('');
        console.log("DEBUG resultHtml:", resultHtml);
        correctedTextDiv.innerHTML = resultHtml;
    
        // --- ISSUES LIST ---
        let issuesHtml = '<ul class="issues-list">';
        sortedCorrections.forEach((correction, index) => {
            issuesHtml += `
                <li class="issue-item ${correction.type}">
                    <div class="issue-header">
                        <span class="issue-type">${correction.type.toUpperCase()}</span>
                        <span class="issue-source">${correction.source || 'Layer 1'}</span>
                        <span class="issue-confidence">Confidence: ${correction.confidence?.toFixed(2)}</span>
                    </div>
                    <div class="issue-content">
                        <div class="original-text">Original: "${correction.original}"</div>
                        <div class="corrected-text">Corrected: "${correction.corrected || correction.suggestion}"</div>
                    </div>
                </li>
            `;
        });
        issuesHtml += '</ul>';
        issuesListDiv.innerHTML = issuesHtml;
    
        addTooltipEvents();
    }
    
    
    

    // REDESIGNED TOOLTIP HANDLING
    let activeTooltip = null;
    let isTooltipPinned = false;

    function addTooltipEvents() {
        const errorSpans = document.querySelectorAll('.error');
        
        errorSpans.forEach(span => {
            span.addEventListener('mouseover', (e) => {
                const correctionIndex = span.getAttribute('data-correction-index');
                if (correctionIndex) {
                    const correction = originalCorrections[correctionIndex];
                    showTooltip(span, correction);
                }
            });

            span.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent document click listener from hiding it
                isTooltipPinned = true;
                if (activeTooltip) {
                    activeTooltip.classList.add('pinned');
                }
            });

            span.addEventListener('mouseout', (e) => {
                // Hide tooltip only if mouse is not over the tooltip and it's not pinned
                if (activeTooltip && !isTooltipPinned && !activeTooltip.matches(':hover')) {
                    hideTooltip();
                }
            });
        });

        // Add a click listener to the document to close pinned tooltips
        document.addEventListener('click', () => {
            if (isTooltipPinned) {
                hideTooltip();
            }
        });
    }

    function showTooltip(span, correction) {
        // Hide any existing tooltip first
        hideTooltip();

        // Create the new tooltip
        activeTooltip = document.createElement('div');
        activeTooltip.className = 'tooltip';
        activeTooltip.innerHTML = `
            <div class="tooltip-header">
                <span class="tooltip-type">${correction.type.toUpperCase()}</span>
                <span class="tooltip-confidence">Confidence: ${correction.confidence.toFixed(2)}</span>
            </div>
            <div class="tooltip-content">
                <div class="text-comparison">
                    <span class="original-label">Original:</span>
                    <span class="original-text">"${correction.original}"</span>
                </div>
                <div class="text-comparison">
                    <span class="corrected-label">Corrected:</span>
                    <span class="corrected-text">"${correction.corrected}"</span>
                </div>
                <div class="tooltip-description">${correction.description || ''}</div>
            </div>
            <div class="tooltip-actions">
                <button class="tooltip-btn reject-btn">Reject Change</button>
            </div>
        `;
        
        document.body.appendChild(activeTooltip);
        positionTooltip(activeTooltip, span);

        // Allow clicking inside the tooltip without closing it
        activeTooltip.addEventListener('click', e => e.stopPropagation());

        // Add event listener to hide tooltip when mouse leaves it
        activeTooltip.addEventListener('mouseleave', () => {
            if (!isTooltipPinned) {
                hideTooltip();
            }
        });

        // Event listeners for action buttons
        const rejectBtn = activeTooltip.querySelector('.reject-btn');

        rejectBtn.addEventListener('click', () => {
            // "Reject" means reverting to the original text
            const originalTextNode = document.createTextNode(correction.original);
            span.parentNode.replaceChild(originalTextNode, span);
            hideTooltip();
        });
    }

    function hideTooltip() {
        if (activeTooltip) {
            activeTooltip.remove();
            activeTooltip = null;
            isTooltipPinned = false;
        }
    }

    function positionTooltip(tooltip, span) {
        const rect = span.getBoundingClientRect();
        tooltip.style.left = `${rect.left + window.scrollX}px`;
        tooltip.style.top = `${rect.bottom + window.scrollY + 5}px`;

        // Adjust if tooltip goes off-screen
        if (rect.bottom + tooltip.offsetHeight > window.innerHeight) {
            tooltip.style.top = `${rect.top + window.scrollY - tooltip.offsetHeight - 5}px`;
        }
    }
    
    function showLoading() {
        // Make sure the loading spinner is inside the corrected text box
        correctedTextDiv.innerHTML = '';
        loadingSpinner.style.display = 'block';
        correctedTextDiv.appendChild(loadingSpinner);
        issuesListDiv.innerHTML = '';
    }
    
    function hideLoading() {
        loadingSpinner.style.display = 'none';
    }
}); 

