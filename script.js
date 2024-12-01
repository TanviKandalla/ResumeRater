document.getElementById("resumeUpload").addEventListener("change", function () {
    const fileName = this.files[0]?.name || "Click to Upload PDF";
    document.querySelector(".upload-label span").textContent = fileName;
});

document.getElementById("scoreButton").addEventListener("click", async (event) => {
    event.preventDefault();

    const fileInput = document.getElementById("resumeUpload");
    const resultDiv = document.getElementById("result");
    const classificationDiv = document.getElementById("classification-result");
    const similarityDiv = document.getElementById("similarity-result"); // Correctly reference the similarity div
    const scoreSpan = document.getElementById("score");

    if (!fileInput.files.length) {
        alert("Please upload a PDF file!");
        return;
    }

    const file = fileInput.files[0];

    if (file.type !== "application/pdf") {
        alert("Only PDF files are allowed!");
        return;
    }

    // Generate a random score (simulate scoring logic)
    const score = Math.floor(Math.random() * 30) + 70; // Score range: 70-100
    scoreSpan.textContent = score;

    // Display the result container
    resultDiv.classList.remove("hidden");

    // Send the file to the backend API for job category classification
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/classify/", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
        }

        const data = await response.json();
        console.log("Backend response:", data); // Debug log

        if (data.error) {
            classificationDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
            return;
        }

        // Display the predicted class index and category
        classificationDiv.innerHTML = `
            <p><strong>Predicted Class Index:</strong> ${data.index}</p>
            <p><strong>Job Category:</strong> ${data.category}</p>
        `;

        // Display similar resumes
        const similarResumesHTML = data.similar_resumes
            .map(resume => `<li>${resume}</li>`)
            .join("");
        similarityDiv.innerHTML = `
            <p><strong>Top 10 Similar Resumes:</strong></p>
            <ul>${similarResumesHTML}</ul>
        `;

    } catch (error) {
        console.error("Error occurred:", error); // Debug the error
        classificationDiv.innerHTML = `<p style="color: red;">An error occurred while classifying the resume. Please try again.</p>`;
    }
});
