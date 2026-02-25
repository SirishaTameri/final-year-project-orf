document.addEventListener('DOMContentLoaded', () => {

    // --- UTILITIES ---
    const showSpinner = (btnId, show) => {
        const btn = document.getElementById(btnId);
        if (!btn) return;
        const spinner = btn.querySelector('.spinner-border');
        if (show) {
            btn.disabled = true;
            spinner?.classList.remove('d-none');
        } else {
            btn.disabled = false;
            spinner?.classList.add('d-none');
        }
    };

    const showError = (msg) => {
        alert("Error: " + msg);
    };

    // --- PREDICT PAGE ---
    const predictForm = document.getElementById('predictionForm');
    if (predictForm) {
        predictForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const text = document.getElementById('jobText').value;
            const model = document.querySelector('input[name="model"]:checked').value;

            if (!text) return;

            showSpinner('predictBtn', true);
            const resultSection = document.getElementById('resultSection');
            resultSection.classList.add('d-none');

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, model })
                });

                const data = await response.json();

                if (!response.ok) throw new Error(data.error || 'Prediction failed');

                // Update UI
                const resultHeader = document.getElementById('predictionResult');
                const confBar = document.getElementById('confidenceBar');
                const confVal = document.getElementById('confidenceValue');
                const alertBox = document.getElementById('recommendationAlert');
                const recText = document.getElementById('recommendationText');

                resultHeader.textContent = data.prediction;
                if (data.is_fraud) {
                    resultHeader.className = 'display-6 fw-bold mb-2 text-danger';
                    confBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-danger';
                    alertBox.className = 'alert alert-danger custom-alert mb-0';
                } else {
                    resultHeader.className = 'display-6 fw-bold mb-2 text-success';
                    confBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-success';
                    alertBox.className = 'alert alert-info custom-alert mb-0';
                }

                const confPercent = (data.confidence * 100).toFixed(1) + '%';
                confBar.style.width = confPercent;
                confVal.textContent = confPercent;
                recText.textContent = data.recommendation;

                document.getElementById('procTime').textContent = data.processing_time.toFixed(3);
                document.getElementById('modelUsed').textContent = data.model;

                resultSection.classList.remove('d-none');

                // Risk Breakdown Mock Logic for Presentation
                const riskBreakdown = document.getElementById('riskBreakdown');
                const riskGrid = document.getElementById('riskGrid');
                if (riskBreakdown && riskGrid) {
                    riskBreakdown.classList.remove('d-none');
                    riskGrid.innerHTML = '';

                    const factors = data.is_fraud ? [
                        { icon: 'fa-envelope-open', title: 'Suspicious Domain', desc: 'Email domain does not match official company records.' },
                        { icon: 'fa-money-bill-wave', title: 'Salary Anomaly', desc: 'Compensation is significantly higher than industry average for this role.' },
                        { icon: 'fa-exclamation-triangle', title: 'High-Pressure Language', desc: 'Urgent calls to action detected in job description.' }
                    ] : [
                        { icon: 'fa-check-circle', title: 'Verified Structure', desc: 'Job posting follows standard professional formatting.' },
                        { icon: 'fa-building', title: 'Company Context', desc: 'Requirements align with historical data for this firm.' },
                        { icon: 'fa-shield-alt', title: 'Safe Content', desc: 'No known fraudulent patterns detected in the text.' }
                    ];

                    factors.forEach(f => {
                        riskGrid.insertAdjacentHTML('beforeend', `
                            <div class="col-md-4">
                                <div class="glass-card p-3 h-100" style="border-radius: 12px; font-size: 0.85rem;">
                                    <i class="fas ${f.icon} mb-2 ${data.is_fraud ? 'text-danger' : 'text-success'}"></i>
                                    <div class="fw-bold text-white mb-1">${f.title}</div>
                                    <div class="text-white-50">${f.desc}</div>
                                </div>
                            </div>
                        `);
                    });
                }

                // Persistence: Save to localStorage history
                const user = JSON.parse(localStorage.getItem('jobguard_user'));
                if (user) {
                    const newScan = {
                        title: text.substring(0, 30) + '...',
                        date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
                        model: data.model.toUpperCase(),
                        prediction: data.is_fraud ? 1 : 0,
                        confidence: data.confidence,
                        location: 'Web Scan'
                    };
                    if (!user.scans) user.scans = [];
                    user.scans.unshift(newScan);
                    // Keep only last 10 scans
                    if (user.scans.length > 10) user.scans.pop();
                    localStorage.setItem('jobguard_user', JSON.stringify(user));
                }

            } catch (err) {
                showError(err.message);
            } finally {
                showSpinner('predictBtn', false);
            }
        });
    }

    // --- COMPARE PAGE ---
    const compareForm = document.getElementById('compareForm');
    if (compareForm) {
        compareForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = document.getElementById('compareText').value;

            if (!text) return;

            showSpinner('compareBtn', true);
            const resultDiv = document.getElementById('compareResult');
            resultDiv.classList.add('d-none');

            try {
                const response = await fetch('/api/compare', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });

                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'Comparison failed');

                // Update BERT
                document.getElementById('bertPred').textContent = data.bert.prediction;
                document.getElementById('bertPred').className = data.bert.prediction === 'FRAUDULENT' ? 'fw-bold text-danger' : 'fw-bold text-success';
                const bertConf = (data.bert.confidence * 100).toFixed(1) + '%';
                document.getElementById('bertConf').textContent = bertConf;
                document.getElementById('bertBar').style.width = bertConf;
                document.getElementById('bertBar').className = `progress-bar ${data.bert.prediction === 'FRAUDULENT' ? 'bg-danger' : 'bg-success'}`;

                // Update RoBERTa
                document.getElementById('robertaPred').textContent = data.roberta.prediction;
                document.getElementById('robertaPred').className = data.roberta.prediction === 'FRAUDULENT' ? 'fw-bold text-danger' : 'fw-bold text-success';
                const robertaConf = (data.roberta.confidence * 100).toFixed(1) + '%';
                document.getElementById('robertaConf').textContent = robertaConf;
                document.getElementById('robertaBar').style.width = robertaConf;
                document.getElementById('robertaBar').className = `progress-bar ${data.roberta.prediction === 'FRAUDULENT' ? 'bg-danger' : 'bg-info'}`; // Keep info/success distinct if legit

                document.getElementById('consensusAnalysis').textContent = data.analysis;
                resultDiv.classList.remove('d-none');

            } catch (err) {
                showError(err.message);
            } finally {
                showSpinner('compareBtn', false);
            }
        });
    }

    // --- BATCH PAGE ---
    const batchForm = document.getElementById('batchForm');
    if (batchForm) {
        batchForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const fileInput = document.getElementById('batchFile');
            const model = document.getElementById('batchModel').value;

            if (!fileInput.files[0]) return;

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('model', model);

            showSpinner('batchBtn', true);
            document.getElementById('batchStats').classList.add('d-none');
            document.getElementById('batchTableContainer').classList.add('d-none');
            const tbody = document.getElementById('batchTableBody');
            tbody.innerHTML = '';

            try {
                const response = await fetch('/api/batch-predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'Batch processing failed');

                // Update Stats
                document.getElementById('totalCount').textContent = data.total_jobs;
                document.getElementById('legitCount').textContent = data.legitimate_count;
                document.getElementById('fraudCount').textContent = data.fraudulent_count;
                document.getElementById('batchTime').textContent = data.processing_time.toFixed(1) + 's';
                document.getElementById('batchStats').classList.remove('d-none');

                // Populate Table
                data.results.forEach(row => {
                    const tr = document.createElement('tr');
                    const badgeClass = row.prediction === 'FRAUDULENT' ? 'badge-fraud' : (row.prediction === 'LEGITIMATE' ? 'badge-legit' : 'badge bg-secondary');

                    tr.innerHTML = `
                        <td class="text-white-50">${row.text}</td>
                        <td><span class="${badgeClass}">${row.prediction}</span></td>
                        <td>${(row.confidence * 100).toFixed(1)}%</td>
                    `;
                    tbody.appendChild(tr);
                });
                document.getElementById('batchTableContainer').classList.remove('d-none');

            } catch (err) {
                showError(err.message);
            } finally {
                showSpinner('batchBtn', false);
            }
        });
    }
});
