// API Configuration
export const API_CONFIG = {
  // Backend URLs
  BACKEND_URL: process.env.BACKEND_URL || 'http://localhost:5000',
  AGENTS_BACKEND_URL: process.env.AGENTS_BACKEND_URL || 'http://localhost:8080',
  
  // API endpoints
  ENDPOINTS: {
    CHAT: '/api/chat',
    CHAT_HISTORY: '/api/chat/history',
    UPLOAD: '/api/upload',
    // Agents endpoints
    AGENTS: {
      HEALTH: '/health',
      CREATE_AGENT: '/api/v1/agents',
      CREATE_ADVANCED_AGENT: '/api/v1/agents/advanced',
      LIST_AGENTS: '/api/v1/agents',
      AGENT_STATUS: '/api/v1/agents',
      INFERENCE: '/api/v1/agents',
      EVALUATE: '/api/v1/agents',
      DELETE_AGENT: '/api/v1/agents',
    },
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
  
  // Build full URL for agents backend requests
  buildAgentsBackendUrl: (endpoint: string) => {
    return `${API_CONFIG.AGENTS_BACKEND_URL}${endpoint}`;
  },
  
  // Build full URL for Next.js API routes
  buildApiUrl: (endpoint: string) => {
    return `/api${endpoint}`;
  },
  
  // Handle API errors
  handleApiError: (error: unknown) => {
    console.error('API Error:', error);
    
    if (error && typeof error === 'object' && 'response' in error) {
      const apiError = error as { response: { data?: { message?: string }; status: number } };
      // Server responded with error status
      return {
        error: apiError.response.data?.message || 'Server error occurred',
        status: apiError.response.status,
      };
    } else if (error && typeof error === 'object' && 'request' in error) {
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
