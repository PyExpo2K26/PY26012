        // Page Navigation
        function switchPage(pageName) {
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.getElementById(pageName).classList.add('active');

            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            event.target.classList.add('active');
        }

        // File Upload
        const uploadBox = document.getElementById('uploadBox');
        const fileInput = document.getElementById('fileInput');

        uploadBox.addEventListener('click', () => fileInput.click());

        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.classList.add('active');
        });

        uploadBox.addEventListener('dragleave', () => {
            uploadBox.classList.remove('active');
        });

        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.classList.remove('active');
            const files = e.dataTransfer.files;
            handleFiles(files);
        });

        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });

        function handleFiles(files) {
            if (files.length > 0) {
                const file = files[0];
                uploadBox.style.opacity = '0.5';
                uploadBox.style.pointerEvents = 'none';

                startAnalysis();

                // Save to history
                saveToHistory(file.name, file.type);
            }
        }

        // Simulated Analysis
        function startAnalysis() {
            let progress = [0, 0, 0, 0];
            const interval = setInterval(() => {
                for (let i = 0; i < 4; i++) {
                    if (progress[i] < 100) {
                        progress[i] += Math.random() * 20;
                        if (progress[i] > 100) progress[i] = 100;
                    }
                }

                document.getElementById('fill1').style.width = progress[0] + '%';
                document.getElementById('fill2').style.width = progress[1] + '%';
                document.getElementById('fill3').style.width = progress[2] + '%';
                document.getElementById('fill4').style.width = progress[3] + '%';

                document.getElementById('prog1').textContent = Math.floor(progress[0]) + '%';
                document.getElementById('prog2').textContent = Math.floor(progress[1]) + '%';
                document.getElementById('prog3').textContent = Math.floor(progress[2]) + '%';
                document.getElementById('prog4').textContent = Math.floor(progress[3]) + '%';

                if (progress.every(p => p === 100)) {
                    clearInterval(interval);
                    showResults();
                }
            }, 300);
        }

        function showResults() {
            document.getElementById('resultsSection').style.display = 'block';
            document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
        }

        // Utility Functions
        function copyToClipboard(btn) {
            const text = btn.previousElementSibling.textContent;
            navigator.clipboard.writeText(text);
            const orig = btn.textContent;
            btn.textContent = 'Copied';
            setTimeout(() => { btn.textContent = orig; }, 2000);
        }

        function downloadReport() {
            alert('Downloading PDF report: forensic_analysis_2026-01-28.pdf');
        }

        function shareResults() {
            alert('Share link copied: https://forgery-detection.app/results/abc123xyz');
        }

        function analyzeAnother() {
            uploadBox.style.opacity = '1';
            uploadBox.style.pointerEvents = 'auto';
            document.getElementById('resultsSection').style.display = 'none';
            [1, 2, 3, 4].forEach(i => {
                document.getElementById('fill' + i).style.width = '0%';
                document.getElementById('prog' + i).textContent = '0%';
            });
        }

        function viewAnalysis() {
            alert('Full analysis details would be shown in detailed view');
        }

        function downloadPDF() {
            alert('Downloading forensic report...');
        }

        function deleteRecord() {
            if (confirm('Delete this analysis?')) {
                event.target.closest('.history-item').remove();
                alert('Deleted');
            }
        }

        function saveToHistory(filename, type) {
            const icon = type.includes('image') ? '' : type.includes('video') ? '' : '';
            const risk = Math.floor(Math.random() * 60 + 30);
            const now = new Date().toLocaleString();

            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `
                <div class="history-info">
                    <h3>${filename}</h3>
                    <p>${type} | Risk: ${risk}% | ${now}</p>
                </div>
                <div class="history-actions">
                    <button class="btn-small" onclick="viewAnalysis()">View</button>
                    <button class="btn-small" onclick="downloadPDF()">PDF</button>
                    <button class="btn-small" onclick="deleteRecord()">Delete</button>
                </div>
            `;
            document.getElementById('historyList').prepend(item);
        }

        // FAQ Toggle
        function toggleFaq(element) {
            const item = element.parentElement;
            const isActive = item.classList.contains('active');

            // Close all other items
            document.querySelectorAll('.faq-item').forEach(i => {
                i.classList.remove('active');
            });

            if (!isActive) {
                item.classList.add('active');
            }
        }

        // Theme Toggle
        function toggleTheme() {
            document.documentElement.style.filter = document.documentElement.style.filter ? '' : 'invert(1) hue-rotate(180deg)';
        }
