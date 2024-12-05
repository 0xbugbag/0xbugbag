"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
class MessageHandler {
    constructor() {
        // Initialize form and status elements
        this.form = document.getElementById('messageForm');
        this.statusDiv = document.getElementById('statusMessage');
        // Bind the submit handler to the form
        this.form.addEventListener('submit', this.handleSubmit.bind(this));
    }
    handleSubmit(event) {
        return __awaiter(this, void 0, void 0, function* () {
            event.preventDefault();
            try {
                // Get form data
                const formData = this.getFormData();
                // Validate form data
                if (!this.validateForm(formData)) {
                    return;
                }
                // Show loading state
                this.showStatus('Sending message...', 'pending');
                // Send the data to your backend
                const response = yield this.sendMessage(formData);
                // Handle the response
                if (response.success) {
                    this.showStatus('Message sent successfully!', 'success');
                    this.form.reset();
                }
                else {
                    this.showStatus('Failed to send message: ' + response.message, 'error');
                }
            }
            catch (error) {
                this.showStatus('An error occurred: ' + error.message, 'error');
            }
        });
    }
    getFormData() {
        return {
            name: document.getElementById('name').value,
            email: document.getElementById('email').value,
            message: document.getElementById('message').value
        };
    }
    validateForm(data) {
        // Add your validation logic here
        if (!data.name.trim()) {
            this.showStatus('Please enter your name', 'error');
            return false;
        }
        if (!data.email.trim() || !this.validateEmail(data.email)) {
            this.showStatus('Please enter a valid email address', 'error');
            return false;
        }
        if (!data.message.trim()) {
            this.showStatus('Please enter a message', 'error');
            return false;
        }
        return true;
    }
    validateEmail(email) {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    }
    sendMessage(data) {
        return __awaiter(this, void 0, void 0, function* () {
            // Replace with your actual API endpoint
            const apiUrl = 'http://localhost:9000/messages';
            const response = yield fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return yield response.json();
        });
    }
    showStatus(message, type) {
        this.statusDiv.textContent = message;
        this.statusDiv.className = `status ${type}`;
    }
}
// Initialize the handler when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MessageHandler();
});
