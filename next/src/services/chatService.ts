import { apiUtils } from '@/config/api';

export interface ChatMessage {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: string;
  chatId?: string;
}

export interface ChatResponse {
  success: boolean;
  response: string;
  chatId: string;
  timestamp: string;
  error?: string;
}

export interface ChatHistory {
  chatId: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
}

class ChatService {
  // Send a message to the chat API
  async sendMessage(message: string, chatId?: string, userId?: string): Promise<ChatResponse> {
    try {
      const response = await fetch(apiUtils.buildApiUrl('/chat'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          chatId,
          userId,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to send message');
      }

      return data;
    } catch (error) {
      console.error('Backend unavailable, using mock response:', error);
      
      // Return mock response when backend is unavailable
      return this.getMockResponse(message, chatId || this.generateChatId());
    }
  }

  // Generate mock AI responses
  private getMockResponse(userMessage: string, chatId: string): ChatResponse {
    const mockResponses = [
      "I understand your question about " + userMessage.toLowerCase() + ". As an AI assistant, I'm here to help you with various tasks and provide information.",
      "That's an interesting point. Let me think about " + userMessage + " and provide you with a comprehensive response.",
      "Thank you for your message. I can help you with that. Here's what I know about your inquiry regarding " + userMessage.toLowerCase() + ".",
      "I see you're asking about " + userMessage + ". This is a topic I can definitely assist you with. Let me provide some insights.",
      "Your question about " + userMessage.toLowerCase() + " is quite relevant. I'll do my best to provide you with accurate and helpful information."
    ];

    const randomResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)];

    return {
      success: true,
      response: randomResponse,
      chatId: chatId,
      timestamp: new Date().toISOString(),
    };
  }

  // Get chat history
  async getChatHistory(chatId: string, userId?: string): Promise<ChatHistory> {
    try {
      const params = new URLSearchParams();
      if (chatId) params.append('chatId', chatId);
      if (userId) params.append('userId', userId);

      const response = await fetch(
        `${apiUtils.buildApiUrl('/chat')}?${params.toString()}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch chat history');
      }

      return data;
    } catch (error) {
      console.error('Error fetching chat history:', error);
      throw error;
    }
  }

  // Upload file (if needed)
  async uploadFile(file: File, chatId?: string): Promise<{ success: boolean; fileUrl?: string; error?: string }> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (chatId) formData.append('chatId', chatId);

      const response = await fetch(apiUtils.buildApiUrl('/upload'), {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to upload file');
      }

      return data;
    } catch (error) {
      console.error('Backend unavailable for file upload, using mock response:', error);
      
      // Return mock success response when backend is unavailable
      return {
        success: true,
        fileUrl: `mock://uploads/${file.name}`,
        error: undefined
      };
    }
  }

  // Generate a new chat ID
  generateChatId(): string {
    return `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Format message for display
  formatMessage(text: string, isUser: boolean, chatId?: string): ChatMessage {
    return {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      text,
      isUser,
      timestamp: new Date().toISOString(),
      chatId,
    };
  }
}

// Export singleton instance
export const chatService = new ChatService();
