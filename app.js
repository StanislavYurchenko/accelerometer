async function runModelInference(X, Y, Z) {
    if (!scalerSession || !classifierSession) return;

    const features = extractFeatures(X, Y, Z);
    featuresDebug.innerText = JSON.stringify(features.map(f => f.toFixed(3)));

    const inputTensor = new ort.Tensor('float32', Float32Array.from(features), [1, 15]);

    try {
        // Step A: Standardization
        const scalerResults = await scalerSession.run({ float_input: inputTensor });
        const scalerOutputKey = Object.keys(scalerResults)[0];
        const scaledFeatures = scalerResults[scalerOutputKey];

        // Step B: Classification (Strictly request ONLY 'label' output from Random Forest)
        const classifierResults = await classifierSession.run(
            { float_input: scaledFeatures },
            ['label']
        );
        
        const predictedClass = classifierResults.label.data[0];
        statusDiv.innerText = activityClasses[predictedClass] || "Unknown class: " + predictedClass;

    } catch (err) {
        console.error("ONNX Runtime Fail:", err);
        statusDiv.innerText = "Error: " + err.message;
    }
}