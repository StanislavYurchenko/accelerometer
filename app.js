const startBtn = document.getElementById('startBtn');
const statusDiv = document.getElementById('status');
const progressSpan = document.getElementById('windowProgress');
const currentMagSpan = document.getElementById('currentMag');
const featuresDebug = document.getElementById('featuresDebug');

let scalerSession = null;
let classifierSession = null;

let windowX = [];
let windowY = [];
let windowZ = [];
const WINDOW_SIZE = 80;

const activityClasses = {
    0: "🧘‍♂️ Rest",
    1: "🚶‍♂️ Walking",
    2: "🧗‍♂️ Stairs",
    3: "🏃‍♂️ Running"
};

// Load ONNX sessions securely using ArrayBuffer (fixes GitHub Pages 404/MIME bugs)
async function initML() {
    try {
        statusDiv.innerText = "Initializing AI...";

        // Fetch scaler.onnx as raw bytes
        const scalerResponse = await fetch('scaler.onnx');
        if (!scalerResponse.ok) throw new Error(`Failed to fetch scaler.onnx (Status: ${scalerResponse.status})`);
        const scalerBuffer = await scalerResponse.arrayBuffer();

        // Fetch classifier.onnx as raw bytes
        const classifierResponse = await fetch('classifier.onnx');
        if (!classifierResponse.ok) throw new Error(`Failed to fetch classifier.onnx (Status: ${classifierResponse.status})`);
        const classifierBuffer = await classifierResponse.arrayBuffer();

        // Initialize ONNX sessions directly from memory buffers
        scalerSession = await ort.InferenceSession.create(new Uint8Array(scalerBuffer));
        classifierSession = await ort.InferenceSession.create(new Uint8Array(classifierBuffer));

        statusDiv.innerText = "Ready to Track";
    } catch (e) {
        // This will show exactly what went wrong during fetch or initialization
        statusDiv.innerText = "ONNX Load Error: " + e.message;
        console.error(e);
    }
}

initML();

startBtn.addEventListener('click', async () => {
    if (typeof DeviceMotionEvent.requestPermission === 'function') {
        try {
            const permissionState = await DeviceMotionEvent.requestPermission();
            if (permissionState === 'granted') {
                startSensorTracking();
            } else {
                alert('Sensor access denied!');
            }
        } catch (error) {
            console.error(error);
        }
    } else {
        startSensorTracking();
    }
});

function startSensorTracking() {
    startBtn.style.display = 'none';
    statusDiv.innerText = "Tracking...";
    
    window.addEventListener('devicemotion', (event) => {
        let x = event.accelerationIncludingGravity.x || 0;
        let y = event.accelerationIncludingGravity.y || 0;
        let z = event.accelerationIncludingGravity.z || 0;

        let currentMag = Math.sqrt(x*x + y*y + z*z);
        currentMagSpan.innerText = currentMag.toFixed(2);

        windowX.push(x);
        windowY.push(y);
        windowZ.push(z);
        
        progressSpan.innerText = windowX.length;

        if (windowX.length >= WINDOW_SIZE) {
            runModelInference(windowX, windowY, windowZ);
            windowX = [];
            windowY = [];
            windowZ = [];
        }
    });
}

// Extract 15 features matching Python logic
function extractFeatures(X, Y, Z) {
    const getMean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
    const getStd = (arr, mean) => Math.sqrt(arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length);
    
    const x_mean = getMean(X);
    const y_mean = getMean(Y);
    const z_mean = getMean(Z);

    const x_std = getStd(X, x_mean);
    const y_std = getStd(Y, y_mean);
    const z_std = getStd(Z, z_mean);

    const x_max = Math.max(...X);
    const y_max = Math.max(...Y);
    const z_max = Math.max(...Z);

    const x_min = Math.min(...X);
    const y_min = Math.min(...Y);
    const z_min = Math.min(...Z);

    let smaSum = 0;
    for (let i = 0; i < WINDOW_SIZE; i++) {
        smaSum += Math.abs(X[i]) + Math.abs(Y[i]) + Math.abs(Z[i]);
    }
    const sma = smaSum / WINDOW_SIZE;

    let magnitudes = [];
    for (let i = 0; i < WINDOW_SIZE; i++) {
        magnitudes.push(Math.sqrt(X[i]**2 + Y[i]**2 + Z[i]**2));
    }
    const mag_mean = getMean(magnitudes);
    const mag_std = getStd(magnitudes, mag_mean);

    return [
        x_mean, y_mean, z_mean,
        x_std, y_std, z_std,
        x_max, y_max, z_max,
        x_min, y_min, z_min,
        sma,
        mag_mean, mag_std
    ];
}

async function runModelInference(X, Y, Z) {
    if (!scalerSession || !classifierSession) return;

    const features = extractFeatures(X, Y, Z);
    
    // Дістаємо точні імена, які зашиті всередині твоїх .onnx файлів
    const scalerInName = scalerSession.inputNames[0];
    const scalerOutName = scalerSession.outputNames[0];
    const classInName = classifierSession.inputNames[0];
    const classOutNames = classifierSession.outputNames; // Тут може бути ['label', 'probabilities']

    // Виводимо цю інфу на екран
    featuresDebug.innerText = `
[DEBUG INFO]
Scaler IN: ${scalerInName} | OUT: ${scalerOutName}
Model IN: ${classInName} | OUTs: ${classOutNames.join(', ')}
-------------------
Features: ${JSON.stringify(features.map(f => f.toFixed(2)))}
    `.trim();

    try {
        const inputTensor = new ort.Tensor('float32', Float32Array.from(features), [1, 15]);

        // Динамічно формуємо вхід для скалера
        let scalerInput = {};
        scalerInput[scalerInName] = inputTensor;
        
        const scalerResults = await scalerSession.run(scalerInput);
        const scaledFeatures = scalerResults[scalerOutName];

        // Динамічно формуємо вхід для моделі
        let classifierInput = {};
        classifierInput[classInName] = scaledFeatures;

        // Строго просимо повернути ТІЛЬКИ перший вихід (зазвичай це label)
        const expectedOutputs = [classOutNames[0]];

        const classifierResults = await classifierSession.run(classifierInput, expectedOutputs);
        
        const predictedClass = classifierResults[classOutNames[0]].data[0];
        statusDiv.innerText = activityClasses[predictedClass] || "Клас: " + predictedClass;

    } catch (err) {
        console.error("ONNX Full Error:", err);
        // Виводимо максимально детальну помилку червоним кольором
        statusDiv.innerHTML = `<span style="font-size: 1.2rem; color: #ef4444;">Помилка: ${err.message}</span>`;
    }
}