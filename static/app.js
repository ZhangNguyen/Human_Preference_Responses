const askBtn = document.getElementById("askBtn");
const questionInput = document.getElementById("question");
const loading = document.getElementById("loading");

const baseAnswer = document.getElementById("baseAnswer");
const ftAnswer = document.getElementById("ftAnswer");
const baseLatency = document.getElementById("baseLatency");
const ftLatency = document.getElementById("ftLatency");

async function askQuestion() {
    const question = questionInput.value.trim();

    if (!question) {
        alert("Vui lòng nhập câu hỏi.");
        return;
    }

    loading.classList.remove("hidden");
    baseAnswer.textContent = "Đang sinh câu trả lời...";
    ftAnswer.textContent = "Đang sinh câu trả lời...";
    baseLatency.textContent = "Thời gian: --";
    ftLatency.textContent = "Thời gian: --";

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question,
                history: []
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Đã xảy ra lỗi.");
        }

        if (data.base_model.error) {
            baseAnswer.textContent = "Lỗi: " + data.base_model.error;
        } else {
            baseAnswer.textContent = data.base_model.answer || "";
        }

        if (data.finetuned_model.error) {
            ftAnswer.textContent = "Lỗi: " + data.finetuned_model.error;
        } else {
            ftAnswer.textContent = data.finetuned_model.answer || "";
        }

        baseLatency.textContent = `Thời gian: ${data.base_model.latency}s`;
        ftLatency.textContent = `Thời gian: ${data.finetuned_model.latency}s`;

    } catch (error) {
        baseAnswer.textContent = "Lỗi: " + error.message;
        ftAnswer.textContent = "Lỗi: " + error.message;
    } finally {
        loading.classList.add("hidden");
    }
}

askBtn.addEventListener("click", askQuestion);

questionInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        askQuestion();
    }
});