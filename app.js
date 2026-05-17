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