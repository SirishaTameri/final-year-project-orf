document.addEventListener('DOMContentLoaded', () => {
    const chatbotToggle = document.getElementById('chatbotToggle');
    const chatbotWindow = document.getElementById('chatbotWindow');
    const closeChatbot = document.getElementById('closeChatbot');
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const chatBody = document.getElementById('chatBody');

    const responses = {
        "hello": "Hello! I'm your JobGuard AI assistant. How can I help you today?",
        "hi": "Hi there! I'm here to help you identify recruitment fraud. What's on your mind?",
        "fraud": "Recruitment fraud often involves urgent language, high salaries for low-skill work, and requests for money or personal data early in the process. Use our 'Single Check' tool to analyze any suspicious postings.",
        "bert": "BERT (Bidirectional Encoder Representations from Transformers) is a deep learning model we use for high-speed, balanced fraud detection.",
        "roberta": "RoBERTa is a robustly optimized BERT approach that provides higher precision for complex job descriptions.",
        "help": "I can help you with: \n1. Identifying common fraud signs\n2. Explaining our AI models\n3. Guidance on using the dashboard",
        "default": "That's an interesting question! For specific job analysis, please use our Single Check or Batch Processing tools. Is there anything else about recruitment fraud I can help with?"
    };

    chatbotToggle.onclick = () => chatbotWindow.classList.toggle('d-none');
    closeChatbot.onclick = () => chatbotWindow.classList.add('d-none');

    chatForm.onsubmit = (e) => {
        e.preventDefault();
        const msg = chatInput.value.trim().toLowerCase();
        if (!msg) return;

        appendMessage('user', chatInput.value);
        chatInput.value = '';

        setTimeout(() => {
            let response = responses.default;
            for (let key in responses) {
                if (msg.includes(key)) {
                    response = responses[key];
                    break;
                }
            }
            appendMessage('bot', response);
        }, 600);
    };

    function appendMessage(sender, text) {
        const div = document.createElement('div');
        div.className = `mb-2 p-2 rounded ${sender === 'user' ? 'bg-primary text-white ms-auto' : 'bg-dark border border-secondary text-white-50'}`;
        div.style.maxWidth = '80%';
        div.style.width = 'fit-content';
        div.innerText = text;
        chatBody.appendChild(div);
        chatBody.scrollTop = chatBody.scrollHeight;
    }
});
