// API Configuration
export const API_CONFIG = {
  // Backend URL - change this to your actual backend URL
  BACKEND_URL: process.env.BACKEND_URL || 'http://localhost:5000',
  
  // API endpoints
  ENDPOINTS: {
    CHAT: '/api/chat',
    CHAT_HISTORY: '/api/chat/history',
    UPLOAD: '/api/upload',
  },
  
  // Request timeout in milliseconds
  TIMEOUT: 30000,
  
  // Default headers
  DEFAULT_HEADERS: {
    'Content-Type': 'application/json',
  },
};

// API utility functions
export const apiUtils = {
  // Build full URL for backend requests
  buildBackendUrl: (endpoint: string) => {
    return `${API_CONFIG.BACKEND_URL}${endpoint}`;
  },
  
  // Build full URL for Next.js API routes
  buildApiUrl: (endpoint: string) => {
    return `/api${endpoint}`;
  },
  
  // Handle API errors
  handleApiError: (error: any) => {
    console.error('API Error:', error);
    
    if (error.response) {
      // Server responded with error status
      return {
        error: error.response.data?.message || 'Server error occurred',
        status: error.response.status,
      };
    } else if (error.request) {
      // Request was made but no response received
      return {
        error: 'Network error - please check your connection',
        status: 0,
      };
    } else {
      // Something else happened
      return {
        error: 'An unexpected error occurred',
        status: 0,
      };
    }
  },
};
