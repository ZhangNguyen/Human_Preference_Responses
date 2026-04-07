const askBtn = document.getElementById("askBtn");
const clearBtn = document.getElementById("clearBtn");
const questionInput = document.getElementById("question");
const loading = document.getElementById("loading");

const title = document.getElementById("title");
const modelName = document.getElementById("modelName");
const latencyText = document.getElementById("latencyText");
const answerBox = document.getElementById("answerBox");
const errorBox = document.getElementById("errorBox");

function setLoading(isLoading) {
    if (isLoading) {
        loading.classList.remove("hidden");
        askBtn.disabled = true;
        askBtn.classList.add("disabled");
    } else {
        loading.classList.add("hidden");
        askBtn.disabled = false;
        askBtn.classList.remove("disabled");
    }
}

function hideError() {
    errorBox.classList.add("hidden");
    errorBox.textContent = "";
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

async function loadHealth() {
    try {
        const response = await fetch("/health");
        if (!response.ok) return;
        title.textContent = "Model gốc đã sẵn sàng để trả lời";
    } catch (err) {
    }
}

async function askQuestion() {
    const question = questionInput.value.trim();

    if (!question) {
        alert("Vui lòng nhập câu hỏi.");
        return;
    }

    hideError();
    setLoading(true);
    answerBox.textContent = "Đang sinh câu trả lời...";
    latencyText.textContent = "--";

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
            throw new Error(data.error || "Đã xảy ra lỗi khi gọi mô hình.");
        }

        if (data.error) {
            showError(data.error);
            answerBox.textContent = "Không thể sinh câu trả lời.";
            return;
        }

        modelName.textContent = `${data.model_name} · ${data.model_dir}`;
        latencyText.textContent = `${data.latency}s`;
        answerBox.textContent = data.answer || "(Không có nội dung trả lời)";
    } catch (error) {
        showError(error.message || "Lỗi không xác định.");
        answerBox.textContent = "Không thể sinh câu trả lời.";
    } finally {
        setLoading(false);
    }
}

askBtn.addEventListener("click", askQuestion);

clearBtn.addEventListener("click", () => {
    questionInput.value = "";
    answerBox.textContent = "Chưa có dữ liệu.";
    latencyText.textContent = "--";
    hideError();
    questionInput.focus();
});

questionInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        askQuestion();
    }
});

loadHealth();