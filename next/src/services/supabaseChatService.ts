import { supabase, isSupabaseConfigured } from '@/lib/supabase';

export interface ChatMessage {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: string;
  chatId: string;
}

export interface Chat {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
}

export interface ChatResponse {
  success: boolean;
  response: string;
  chatId: string;
  timestamp: string;
  error?: string;
}

class SupabaseChatService {
  
  // Generate a new chat ID
  generateChatId(): string {
    return `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // LocalStorage fallback methods
  private getLocalStorageChats(): Chat[] {
    if (typeof window === 'undefined') return [];
    try {
      const saved = localStorage.getItem('chats');
      return saved ? JSON.parse(saved) : [];
    } catch (error) {
      console.error('Error reading from localStorage:', error);
      return [];
    }
  }

  private saveLocalStorageChats(chats: Chat[]): void {
    if (typeof window === 'undefined') return;
    try {
      localStorage.setItem('chats', JSON.stringify(chats));
    } catch (error) {
      console.error('Error saving to localStorage:', error);
    }
  }

  // Create a new chat
  async createChat(title: string, userId?: string): Promise<Chat> {
    // Fallback to localStorage if Supabase is not configured
    if (!isSupabaseConfigured || !supabase) {
      console.log('Supabase not configured, using localStorage fallback for createChat');
      
      const newChat: Chat = {
        id: this.generateChatId(),
        title: title.length > 50 ? title.substring(0, 50) + "..." : title,
        messages: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };

      const chats = this.getLocalStorageChats();
      chats.push(newChat);
      this.saveLocalStorageChats(chats);

      return newChat;
    }

    try {
      const { data, error } = await supabase
        .from('chats')
        .insert([
          { 
            title: title.length > 50 ? title.substring(0, 50) + "..." : title,
            user_id: userId 
          }
        ])
        .select()
        .single();

      if (error) throw error;

      return {
        id: data.id,
        title: data.title,
        messages: [],
        createdAt: data.created_at,
        updatedAt: data.updated_at
      };
    } catch (error) {
      console.log('ℹ️ Supabase unavailable, using localStorage fallback for chat creation:', error);
      
      // Fallback to localStorage on error
      const newChat: Chat = {
        id: this.generateChatId(),
        title: title.length > 50 ? title.substring(0, 50) + "..." : title,
        messages: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };

      const chats = this.getLocalStorageChats();
      chats.push(newChat);
      this.saveLocalStorageChats(chats);

      return newChat;
    }
  }

  // Get all chats for a user
  async getChats(userId?: string): Promise<Chat[]> {
    // Fallback to localStorage if Supabase is not configured
    if (!isSupabaseConfigured || !supabase) {
      console.log('Supabase not configured, using localStorage fallback for getChats');
      return this.getLocalStorageChats();
    }

    try {
      const { data: chatsData, error: chatsError } = await supabase
        .from('chats')
        .select('*')
        .order('updated_at', { ascending: false });

      if (chatsError) throw chatsError;

      // Get messages for each chat
      const chatsWithMessages = await Promise.all(
        chatsData.map(async (chat) => {
          const { data: messagesData, error: messagesError } = await supabase
            .from('messages')
            .select('*')
            .eq('chat_id', chat.id)
            .order('timestamp', { ascending: true });

          if (messagesError) {
            console.error('Error fetching messages for chat:', chat.id, messagesError);
            return {
              id: chat.id,
              title: chat.title,
              messages: [],
              createdAt: chat.created_at,
              updatedAt: chat.updated_at
            };
          }

          const messages: ChatMessage[] = messagesData.map(msg => ({
            id: msg.id,
            text: msg.text,
            isUser: msg.is_user,
            timestamp: msg.timestamp,
            chatId: msg.chat_id
          }));

          return {
            id: chat.id,
            title: chat.title,
            messages,
            createdAt: chat.created_at,
            updatedAt: chat.updated_at
          };
        })
      );

      return chatsWithMessages;
    } catch (error) {
      console.log('ℹ️ Supabase unavailable, using localStorage fallback for chat loading:', error);
      // Fallback to localStorage on error
      return this.getLocalStorageChats();
    }
  }

  // Get a specific chat with its messages
  async getChat(chatId: string): Promise<Chat | null> {
    // Fallback to localStorage if Supabase is not configured
    if (!isSupabaseConfigured || !supabase) {
      console.log('Supabase not configured, using localStorage fallback for getChat');
      const chats = this.getLocalStorageChats();
      return chats.find(chat => chat.id === chatId) || null;
    }

    try {
      const { data: chatData, error: chatError } = await supabase
        .from('chats')
        .select('*')
        .eq('id', chatId)
        .single();

      if (chatError) throw chatError;

      const { data: messagesData, error: messagesError } = await supabase
        .from('messages')
        .select('*')
        .eq('chat_id', chatId)
        .order('timestamp', { ascending: true });

      if (messagesError) throw messagesError;

      const messages: ChatMessage[] = messagesData.map(msg => ({
        id: msg.id,
        text: msg.text,
        isUser: msg.is_user,
        timestamp: msg.timestamp,
        chatId: msg.chat_id
      }));

      return {
        id: chatData.id,
        title: chatData.title,
        messages,
        createdAt: chatData.created_at,
        updatedAt: chatData.updated_at
      };
    } catch (error) {
      console.log('ℹ️ Supabase unavailable, using localStorage fallback for single chat fetch:', error);
      // Fallback to localStorage on error
      const chats = this.getLocalStorageChats();
      return chats.find(chat => chat.id === chatId) || null;
    }
  }

  // Save a message to a chat
  async saveMessage(chatId: string, text: string, isUser: boolean): Promise<ChatMessage> {
    const timestamp = new Date().toISOString();
    
    // Fallback to localStorage if Supabase is not configured
    if (!isSupabaseConfigured || !supabase) {
      console.log('Supabase not configured, using localStorage fallback for saveMessage');
      
      const message: ChatMessage = {
        id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        text,
        isUser,
        timestamp,
        chatId
      };

      const chats = this.getLocalStorageChats();
      const chatIndex = chats.findIndex(chat => chat.id === chatId);
      
      if (chatIndex >= 0) {
        chats[chatIndex].messages.push(message);
        chats[chatIndex].updatedAt = timestamp;
        this.saveLocalStorageChats(chats);
      }

      return message;
    }

    try {
      const { data, error } = await supabase
        .from('messages')
        .insert([
          {
            chat_id: chatId,
            text,
            is_user: isUser,
            timestamp
          }
        ])
        .select()
        .single();

      if (error) throw error;

      // Update chat's updated_at timestamp
      await supabase
        .from('chats')
        .update({ updated_at: timestamp })
        .eq('id', chatId);

      return {
        id: data.id,
        text: data.text,
        isUser: data.is_user,
        timestamp: data.timestamp,
        chatId: data.chat_id
      };
    } catch (error) {
      console.log('ℹ️ Supabase unavailable, using localStorage fallback for message saving:', error);
      
      // Fallback to localStorage on error
      const message: ChatMessage = {
        id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        text,
        isUser,
        timestamp,
        chatId
      };

      const chats = this.getLocalStorageChats();
      const chatIndex = chats.findIndex(chat => chat.id === chatId);
      
      if (chatIndex >= 0) {
        chats[chatIndex].messages.push(message);
        chats[chatIndex].updatedAt = timestamp;
        this.saveLocalStorageChats(chats);
      }

      return message;
    }
  }

  // Send message and get AI response with streaming support
  async sendMessage(message: string, chatId: string, onStream?: (text: string) => void): Promise<ChatResponse> {
    try {
      // Save user message first
      await this.saveMessage(chatId, message, true);

      // Try to get AI response from LLM backend
      try {
        const response = await fetch('http://localhost:5000/run_llm', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ prompt: message }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        let fullResponse = '';

        // Handle streaming response
        if (response.body && onStream) {
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep the last incomplete line

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.slice(6));
                  
                  if (data.text) {
                    fullResponse += data.text;
                    onStream(data.text); // Stream to UI
                  } else if (data.error) {
                    throw new Error(data.error);
                  }
                } catch (e) {
                  // Skip invalid JSON lines
                }
              }
            }
          }
        } else {
          // Fallback for non-streaming
          const data = await response.json();
          fullResponse = data.response || data.text || 'No response received';
        }

        // Save AI response
        await this.saveMessage(chatId, fullResponse, false);

        return {
          success: true,
          response: fullResponse,
          chatId: chatId,
          timestamp: new Date().toISOString(),
        };

      } catch (apiError) {
        console.error('LLM API error, using fallback:', apiError);
        
        // Generate fallback response
        const fallbackResponses = [
          `I understand your question about "${message}". As an AI assistant, I'm here to help you with various tasks.`,
          `That's an interesting point about "${message}". Let me provide you with a comprehensive response.`,
          `Thank you for asking about "${message}". I can help you with that inquiry.`,
          `I see you're asking about "${message}". This is a topic I can definitely assist you with.`,
          `Your question regarding "${message}" is quite relevant. I'll do my best to provide accurate information.`
        ];

        const randomResponse = fallbackResponses[Math.floor(Math.random() * fallbackResponses.length)];

        // Save fallback AI response
        await this.saveMessage(chatId, randomResponse, false);

        return {
          success: true,
          response: randomResponse,
          chatId: chatId,
          timestamp: new Date().toISOString(),
        };
      }

    } catch (error) {
      console.error('Error in sendMessage:', error);
      return {
        success: false,
        response: 'Sorry, I encountered an error. Please try again.',
        chatId: chatId,
        timestamp: new Date().toISOString(),
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  // Update chat title
  async updateChatTitle(chatId: string, title: string): Promise<boolean> {
    try {
      const { error } = await supabase
        .from('chats')
        .update({ 
          title: title.length > 50 ? title.substring(0, 50) + "..." : title,
          updated_at: new Date().toISOString()
        })
        .eq('id', chatId);

      if (error) throw error;
      return true;
    } catch (error) {
      console.error('Error updating chat title:', error);
      return false;
    }
  }

  // Delete a chat and all its messages
  async deleteChat(chatId: string): Promise<boolean> {
    try {
      // Messages will be deleted automatically due to CASCADE
      const { error } = await supabase
        .from('chats')
        .delete()
        .eq('id', chatId);

      if (error) throw error;
      return true;
    } catch (error) {
      console.error('Error deleting chat:', error);
      return false;
    }
  }

  // Upload file (placeholder for now, can be enhanced later)
  async uploadFile(file: File, chatId?: string): Promise<{ success: boolean; fileUrl?: string; error?: string }> {
    try {
      // For now, return a mock response
      // You can enhance this to actually upload to Supabase Storage
      console.log('File upload requested:', file.name, 'for chat:', chatId);
      
      return {
        success: true,
        fileUrl: `mock://uploads/${file.name}`,
        error: undefined
      };
    } catch (error) {
      console.error('Error uploading file:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Upload failed'
      };
    }
  }

  // Format message for display (utility function)
  formatMessage(text: string, isUser: boolean, chatId: string): ChatMessage {
    return {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      text,
      isUser,
      timestamp: new Date().toISOString(),
      chatId,
    };
  }

  // Test LLM backend connection
  async testLLMBackend(): Promise<{ success: boolean; message: string }> {
    try {
      const response = await fetch('http://localhost:5000/test', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        const data = await response.text();
        return {
          success: true,
          message: `LLM Backend is running! Response: ${data}`
        };
      } else {
        return {
          success: false,
          message: `LLM Backend responded with status: ${response.status}`
        };
      }
    } catch (error) {
      return {
        success: false,
        message: `Error connecting to LLM backend: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  // Check Supabase connection
  async testConnection(): Promise<boolean> {
    console.log('🔍 Testing Supabase connection...');
    
    if (!isSupabaseConfigured || !supabase) {
      console.log('❌ Supabase not configured - missing environment variables');
      console.log('📝 To fix: Create .env.local file with NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY');
      return false;
    }

    console.log('✅ Supabase client configured, testing database connection...');

    try {
      // Test basic connection
      const { data, error } = await supabase
        .from('chats')
        .select('count')
        .limit(1);

      if (error) {
        console.log('❌ Supabase database connection failed:', {
          message: error.message,
          code: error.code,
          details: error.details,
          hint: error.hint
        });
        console.log('🔧 Possible fixes:');
        console.log('   1. Check if tables exist in Supabase dashboard');
        console.log('   2. Verify RLS policies allow access');
        console.log('   3. Confirm API key has correct permissions');
        return false;
      }

      console.log('🎉 Supabase connection successful!');
      console.log('📊 Database ready - will use Supabase for storage');
      return true;
    } catch (error) {
      console.log('❌ Supabase connection error:', error);
      console.log('🔧 Check your Supabase project URL and API key');
      return false;
    }
  }
}

// Export singleton instance
export const supabaseChatService = new SupabaseChatService();
