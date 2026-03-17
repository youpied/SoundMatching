from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
import io
import os
import base64
import matplotlib.pyplot as plt

app = FastAPI()

# 31バンドEQ周波数
eq_freqs = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100,
    125, 160, 200, 250, 315, 400, 500, 630,
    800, 1000, 1250, 1600, 2000, 2500, 3150,
    4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
])

def smooth_spectrum(spec):
    return uniform_filter1d(spec, size=15)

def calculate_eq_logic(clean, recorded, sr):
    n_fft = 4096
    
    clean_spec = np.abs(librosa.stft(clean, n_fft=n_fft))
    rec_spec = np.abs(librosa.stft(recorded, n_fft=n_fft))

    clean_avg = np.mean(clean_spec, axis=1)
    rec_avg = np.mean(rec_spec, axis=1)

    clean_db = librosa.amplitude_to_db(clean_avg)
    rec_db = librosa.amplitude_to_db(rec_avg)

    clean_db = smooth_spectrum(clean_db)
    rec_db = smooth_spectrum(rec_db)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    eq_values = []

    for f in eq_freqs:
        low = f / np.sqrt(2)
        high = f * np.sqrt(2)
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if len(idx) > 0:
            diff = np.mean(rec_db[idx] - clean_db[idx])
        else:
            diff = 0
        eq_values.append(diff)

    eq_values = np.array(eq_values)
    eq_values *= 0.7
    eq_values -= np.mean(eq_values)
    eq_values = np.clip(eq_values, -20, 20)
    
    return np.round(eq_values).astype(int)

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SoundMatch EQ | Audio Character Transfer</title>
    <meta name="description" content="Match the frequency response of your audio files with AI-powered EQ generation.">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #09090b;
            --surface: #18181b;
            --surface-hover: #27272a;
            --primary: #3b82f6;
            --primary-glow: rgba(59, 130, 246, 0.5);
            --accent: #8b5cf6;
            --text: #fafafa;
            --text-muted: #a1a1aa;
            --border: rgba(255, 255, 255, 0.1);
            --glass: rgba(24, 24, 27, 0.7);
        }

        * {
            box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.15) 0px, transparent 50%);
            color: var(--text);
            margin: 0;
            line-height: 1.5;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .app-container {
            width: 100%;
            max-width: 900px;
            padding: 2rem 1rem;
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            text-align: center;
            margin-bottom: 3rem;
        }

        h1 {
            font-size: clamp(2.5rem, 8vw, 3.5rem);
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0;
            background: linear-gradient(135deg, #fff 30%, #a1a1aa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }

        .main-card {
            background: var(--glass);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 2rem;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .upload-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        @media (max-width: 640px) {
            .upload-grid {
                grid-template-columns: 1fr;
            }
            .main-card {
                padding: 1.5rem;
            }
        }

        .drop-zone {
            position: relative;
            background: rgba(255, 255, 255, 0.02);
            border: 2px dashed var(--border);
            border-radius: 1.25rem;
            padding: 2rem 1.5rem;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            overflow: hidden;
        }

        .drop-zone:hover, .drop-zone.drag-over {
            border-color: var(--primary);
            background: rgba(59, 130, 246, 0.05);
            transform: translateY(-2px);
        }

        .drop-zone.active {
            border-color: var(--accent);
            background: rgba(139, 92, 246, 0.05);
        }

        .drop-zone input {
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
        }

        .icon-circle {
            width: 48px;
            height: 48px;
            background: var(--surface);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1rem;
            border: 1px solid var(--border);
            transition: transform 0.3s ease;
        }

        .drop-zone:hover .icon-circle {
            transform: scale(1.1) rotate(5deg);
        }

        .file-label {
            display: block;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }

        .file-name {
            font-size: 0.875rem;
            color: var(--text-muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 100%;
        }

        .btn-primary {
            width: 100%;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 1rem;
            padding: 1.25rem;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 15px -3px var(--primary-glow);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .btn-primary:hover {
            filter: brightness(1.1);
            transform: translateY(-2px);
            box-shadow: 0 20px 25px -5px var(--primary-glow);
        }

        .btn-primary:active {
            transform: translateY(0);
        }

        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        #result-area {
            margin-top: 3rem;
            display: none;
            animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
        }

        .graph-wrapper {
            background: rgba(0,0,0,0.3);
            border-radius: 1.5rem;
            padding: 1rem;
            border: 1px solid var(--border);
            margin-bottom: 2rem;
        }

        #graph-img {
            width: 100%;
            height: auto;
            border-radius: 1rem;
            filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.2));
        }

        .eq-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 0.75rem;
        }

        .eq-chip {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 0.75rem;
            text-align: center;
            transition: all 0.2s ease;
        }

        .eq-chip:hover {
            border-color: var(--primary);
            background: var(--surface-hover);
        }

        .eq-f {
            display: block;
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
        }

        .eq-v {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--primary);
        }

        .loader {
            display: none;
            margin: 2rem auto;
            text-align: center;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid rgba(255,255,255,0.1);
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            animation: rotate 1s linear infinite;
            margin: 0 auto 1rem;
        }

        @keyframes rotate {
            to { transform: rotate(360deg); }
        }

        footer {
            margin-top: auto;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.875rem;
            text-align: center;
        }

        /* Accessibility: focus states */
        .drop-zone:focus-within {
            outline: 2px solid var(--primary);
            outline-offset: 4px;
        }

        /* Animations for values */
        .eq-v.positive { color: #10b981; }
        .eq-v.negative { color: #ef4444; }

    </style>
</head>
<body>
    <div class="app-container">
        <header>
            <h1>SoundMatch EQ</h1>
            <p class="subtitle">Transfer frequency characteristics between tracks</p>
        </header>

        <main class="main-card">
            <div class="upload-grid">
                <!-- Reference File -->
                <div class="drop-zone" id="zone-clean" role="button" aria-label="Upload original audio">
                    <div class="icon-circle">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M2 12h20"/></svg>
                    </div>
                    <span class="file-label">Original Audio</span>
                    <span class="file-name" id="name-clean">Target frequency response</span>
                    <input type="file" id="clean_file" accept="audio/*, .mp3, .wav, .m4a, .aac, .flac, .ogg">
                </div>

                <!-- Recorded File -->
                <div class="drop-zone" id="zone-recorded" role="button" aria-label="Upload recorded audio">
                    <div class="icon-circle">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M2 12h20"/></svg>
                    </div>
                    <span class="file-label">Recorded Audio</span>
                    <span class="file-name" id="name-recorded">Audio to be corrected</span>
                    <input type="file" id="recorded_file" accept="audio/*, .mp3, .wav, .m4a, .aac, .flac, .ogg">
                </div>
            </div>

            <button class="btn-primary" id="btn-generate" onclick="generateEQ()">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 11a9 9 0 0 1 8-7 9 9 0 0 1 8 7"/><path d="M4 13a9 9 0 0 0 8 7 9 9 0 0 0 8-7"/><polyline points="12 4 12 12 16 16"/></svg>
                Generate Correction Curve
            </button>

            <div class="loader" id="loader">
                <div class="spinner"></div>
                <p>Analyzing sonic fingerprints...</p>
            </div>
        </main>

        <section id="result-area">
            <div class="result-header">
                <h2 style="margin:0; font-size:1.5rem">Correction Curve</h2>
                <div id="status-badge" style="background:var(--primary); padding:4px 12px; border-radius:99px; font-size:0.75rem; font-weight:700">READY</div>
            </div>

            <div class="graph-wrapper">
                <img id="graph-img" src="" alt="EQ Frequency Response Graph">
            </div>

            <h3 style="margin-top:2rem; font-size:1.1rem; color:var(--text-muted)">Equalizer Settings (31-Band)</h3>
            <div class="eq-grid" id="eq-grid"></div>
        </section>
    </div>

    <footer>
        <p>&copy; 2026 SoundMatch EQ. Professional Audio Analysis Tool.</p>
    </footer>

    <script>
        const cleanInput = document.getElementById('clean_file');
        const recordedInput = document.getElementById('recorded_file');

        // UI Updates for file selection
        cleanInput.addEventListener('change', e => {
            if(e.target.files[0]) {
                document.getElementById('name-clean').textContent = e.target.files[0].name;
                document.getElementById('zone-clean').classList.add('active');
            }
        });

        recordedInput.addEventListener('change', e => {
            if(e.target.files[0]) {
                document.getElementById('name-recorded').textContent = e.target.files[0].name;
                document.getElementById('zone-recorded').classList.add('active');
            }
        });

        // Drag and drop visual feedback
        ['zone-clean', 'zone-recorded'].forEach(id => {
            const zone = document.getElementById(id);
            zone.addEventListener('dragover', () => zone.classList.add('drag-over'));
            zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
            zone.addEventListener('drop', () => zone.classList.remove('drag-over'));
        });

        async function generateEQ() {
            const cleanFile = cleanInput.files[0];
            const recordedFile = recordedInput.files[0];

            if (!cleanFile || !recordedFile) {
                alert("Please select both files to continue.");
                return;
            }

            const formData = new FormData();
            formData.append('clean', cleanFile);
            formData.append('recorded', recordedFile);

            const loader = document.getElementById('loader');
            const resultArea = document.getElementById('result-area');
            const btn = document.getElementById('btn-generate');

            loader.style.display = 'block';
            resultArea.style.display = 'none';
            btn.disabled = true;

            try {
                const response = await fetch('/calculate', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error("Sonic analysis failed. Please try different files.");

                const data = await response.json();
                
                // Populate EQ Grid
                const grid = document.getElementById('eq-grid');
                grid.innerHTML = '';
                data.eq_values.forEach((v, i) => {
                    const freq = data.freqs[i];
                    const valClass = v > 0 ? 'positive' : (v < 0 ? 'negative' : '');
                    const div = document.createElement('div');
                    div.className = 'eq-chip';
                    div.innerHTML = `
                        <span class="eq-f">${freq >= 1000 ? (freq/1000).toFixed(1)+'k' : freq}Hz</span>
                        <span class="eq-v ${valClass}">${v > 0 ? '+' : ''}${v}</span>
                    `;
                    grid.appendChild(div);
                });

                document.getElementById('graph-img').src = 'data:image/png;base64,' + data.graph;
                resultArea.style.display = 'block';
                
                // Smooth scroll to results
                resultArea.scrollIntoView({ behavior: 'smooth', block: 'start' });

            } catch (err) {
                alert(err.message);
            } finally {
                loader.style.display = 'none';
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>
    """

@app.post("/calculate")
async def calculate_eq(clean: UploadFile = File(...), recorded: UploadFile = File(...)):
    try:
        # Load audio files
        clean_bytes = await clean.read()
        recorded_bytes = await recorded.read()
        
        # We need soundfile or audioread to load from bytes
        # librosa.load can take a file object
        clean_io = io.BytesIO(clean_bytes)
        recorded_io = io.BytesIO(recorded_bytes)
        
        y_clean, sr_clean = librosa.load(clean_io, sr=None)
        y_recorded, sr_recorded = librosa.load(recorded_io, sr=None)
        
        if sr_clean != sr_recorded:
            # Resample recorded to clean sr if they are different
            y_recorded = librosa.resample(y_recorded, orig_sr=sr_recorded, target_sr=sr_clean)
            sr_recorded = sr_clean

        eq_values = calculate_eq_logic(y_clean, y_recorded, sr_clean)
        
        # Generate Plot
        plt.figure(figsize=(10, 5))
        plt.style.use('dark_background')
        plt.plot(eq_freqs, eq_values, marker='o', color='#0071e3', linewidth=2)
        plt.xscale('log')
        plt.xticks(eq_freqs[::3], [f"{f}" for f in eq_freqs[::3]])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain (dB)')
        plt.title('Generated EQ Curve')
        plt.grid(True, alpha=0.2)
        
        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return {
            "eq_values": eq_values.tolist(),
            "freqs": eq_freqs.tolist(),
            "graph": graph_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
