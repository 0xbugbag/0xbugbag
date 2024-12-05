import { config } from './config';

// Define an interface for the form data
interface MessageForm {
    name: string;
    email: string;
    message: string;
}

// Define an interface for the API response
interface ApiResponse {
    success: boolean;
    message: string;
}

class MessageHandler {
    private form: HTMLFormElement;
    private statusDiv: HTMLDivElement;

    constructor() {
        // Initialize form and status elements
        this.form = document.getElementById('messageForm') as HTMLFormElement;
        this.statusDiv = document.getElementById('statusMessage') as HTMLDivElement;

        // Bind the submit handler to the form
        if (this.form) {
        this.form.addEventListener('submit', this.handleSubmit.bind(this));
        }
    }

    private async handleSubmit(event: Event): Promise<void> {
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
            const response = await this.sendMessage(formData);

            // Handle the response
            if (response.success) {
                this.showStatus('Message sent successfully!', 'success');
                this.form.reset();
            } else {
                this.showStatus('Failed to send message: ' + response.message, 'error');
            }

        } catch (error) {
            this.showStatus('An error occurred: ' + (error as Error).message, 'error');
        }
    }

    private async sendMessage(data: MessageForm): Promise<ApiResponse> {
        const response = await fetch(`${config.apiUrl}/messages`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
            credentials: 'include' // If you need to handle cookies
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json() as ApiResponse;
    }

    private getFormData(): MessageForm {
        return {
            name: (document.getElementById('name') as HTMLInputElement).value,
            email: (document.getElementById('email') as HTMLInputElement).value,
            message: (document.getElementById('message') as HTMLTextAreaElement).value
        };
    }

    private validateForm(data: MessageForm): boolean {
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

    private validateEmail(email: string): boolean {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    }

    private async sendMessage(data: MessageForm): Promise<ApiResponse> {
        // Replace with your actual API endpoint
        const apiUrl = 'https://scala-backend.onrender.com/messages';

        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json() as ApiResponse;
    }

    private showStatus(message: string, type: 'success' | 'error' | 'pending'): void {
        this.statusDiv.textContent = message;
        this.statusDiv.className = `status ${type}`;
    }
}

export default MessageHandler;

// Initialize the handler when the DOM is loaded
//document.addEventListener('DOMContentLoaded', () => {
//    new MessageHandler();
//});
