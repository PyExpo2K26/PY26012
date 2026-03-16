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
