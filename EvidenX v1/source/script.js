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
