async function uploadFile() {
    const fileInput = document.getElementById("file");
    const status = document.getElementById("status");
    const tempoEl = document.getElementById("tempo");
    const freqEl = document.getElementById("frequency");

    if (!fileInput || !fileInput.files.length) {
        alert("Please select an MP3 file");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    status.innerText = "🧬 Analyzing audio...";

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        // handle backend error safely
        if (data.error) {
            status.innerText = "❌ Server error";
            console.error(data.error);
            return;
        }

        // update UI safely
        if (tempoEl) tempoEl.innerText = data.tempo ?? "--";
        if (freqEl) freqEl.innerText = data.frequency ?? "--";

        status.innerText = "✅ Analysis complete";

        // simulate growth curve if available
        if (data.growth_curve && typeof drawGrowthCurve === "function") {
            drawGrowthCurve(data.growth_curve);
        }

    } catch (err) {
        console.error(err);
        status.innerText = "❌ Error connecting to server";
    }
}