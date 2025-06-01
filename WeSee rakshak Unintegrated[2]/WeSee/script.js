document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const voiceBtn = document.getElementById('voice-btn');
    const voiceAnimation = document.getElementById('voice-animation');

    // Function to add a message to the chat
    function addMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = content;
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Handle text input
    function handleTextInput() {
        const message = userInput.value.trim();
        if (message) {
            addMessage(message, true);
            userInput.value = '';
            // Here you would typically send the message to your backend
            // For now, we'll just echo it back
            setTimeout(() => {
                addMessage("I received your message: " + message);
            }, 1000);
        }
    }

    // Handle voice input
    function handleVoiceInput() {
        voiceAnimation.classList.add('active');
        // Here you would typically start voice recognition
        // For now, we'll just simulate it
        setTimeout(() => {
            voiceAnimation.classList.remove('active');
            addMessage("Voice input received!", true);
            setTimeout(() => {
                addMessage("I heard your voice message!");
            }, 1000);
        }, 2000);
    }

    // Event listeners
    sendBtn.addEventListener('click', handleTextInput);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleTextInput();
        }
    });
    voiceBtn.addEventListener('click', handleVoiceInput);
}); 